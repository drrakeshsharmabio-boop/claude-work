"""make_train_val_split.py - leakage-safe train / validation split for page-level KG extraction data.

Why not a random split: every prompt carries the closing text of the PREVIOUS page ("PREVIOUS PAGE
(page N-1) - continuity context only"). If page 1003 is in validation and 1004 in training, the
training prompt for 1004 contains validation text; neighbouring pages also share topics, figures and
tables. A random split therefore makes validation scores look better than they are.

This script groups consecutive pages of a document into clusters and assigns WHOLE clusters to
validation until the requested share of rows is reached. A run of consecutive pages longer than
--block pages (e.g. a complete book) is cut into blocks of at most --block pages; the training pages
that touch a validation block are removed from training ("buffer", written to buffer.jsonl) so no
validation page has a neighbour in training. Optionally whole documents can be held out
(--holdout-docs) to measure transfer to a book the model has never seen.

Input formats (--format):
  vertex   Vertex AI batch output (.jsonl): {"request": {...prompt...}, "response": {...}}; the
           document name and page number are read from "Focus ONLY on PAGE N of source document ...".
  dataset  Training rows ({"raw_ocr_text", "extracted_graph", ...}); the page and the PDF name are
           read from the row, or from the extracted_graph JSON, like the v2 notebook does.
  chunk    Extraction chunks ({"pdf_name", "page", "prompt"}).

Outputs (written to --out): train.jsonl[.gz], val.jsonl[.gz], buffer.jsonl[.gz] (rows unchanged),
split_report.json.

Example:
  python make_train_val_split.py --input predictions.jsonl --format vertex --out split_v1 --val-frac 0.10
"""
import argparse
import collections
import gzip
import json
import os
import random
import re
import statistics as st

PAGE_KEYS = ("page", "page_num", "page_number", "_page")
PDF_KEYS = ("pdf_file", "pdf_name", "source_pdf", "pdf", "source", "book")
FOCUS_RE = re.compile(r'Focus ONLY on PAGE (\d+) of source document "([^"]+)"')


def opener(path, mode):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def vertex_prompt(row):
    return "".join(p.get("text", "") for p in row["request"]["contents"][0]["parts"])


def doc_and_page(row, fmt):
    """Return (document, page) or (None, None)."""
    if fmt == "vertex":
        m = FOCUS_RE.search(vertex_prompt(row))
        return (m.group(2), int(m.group(1))) if m else (None, None)
    if fmt == "chunk":
        if row.get("pdf_name") is None or row.get("page") is None:
            m = FOCUS_RE.search(row.get("prompt", ""))
            return (m.group(2), int(m.group(1))) if m else (None, None)
        return row["pdf_name"], int(row["page"])
    # dataset
    graph = {}
    try:
        g = json.loads(row["extracted_graph"])
        graph = g if isinstance(g, dict) else {}
    except Exception:
        pass
    page = next((int(row[k]) for k in PAGE_KEYS if row.get(k) not in (None, "")), None)
    if page is None and graph.get("page") not in (None, ""):
        page = int(graph["page"])
    if page is None:
        for e in graph.get("entities", []):
            if isinstance(e, dict) and e.get("_page", e.get("page")) is not None:
                page = int(e.get("_page", e.get("page")))
                break
    doc = next((os.path.basename(str(row[k])) for k in PDF_KEYS if row.get(k)), None) or graph.get("pdf_file")
    return (doc, page) if doc and page is not None else (None, None)


def graph_of(row, fmt):
    try:
        if fmt == "vertex":
            return json.loads(row["response"]["candidates"][0]["content"]["parts"][0]["text"])
        if fmt == "dataset":
            return json.loads(row["extracted_graph"])
    except Exception:
        pass
    return None


def build_clusters(keys, gap, block):
    """keys: list of (doc, page, row_index). Pages of one doc within `gap` of each other share a
    cluster; runs longer than `block` rows are cut into consecutive blocks of at most `block`."""
    by_doc = collections.defaultdict(list)
    for doc, page, idx in keys:
        by_doc[doc].append((page, idx))
    clusters = []

    def flush(doc, run):
        for i in range(0, len(run), block):
            clusters.append((doc, run[i:i + block]))

    for doc, items in by_doc.items():
        items.sort()
        cur = [items[0]]
        for page, idx in items[1:]:
            if page - cur[-1][0] <= gap:
                cur.append((page, idx))
            else:
                flush(doc, cur)
                cur = [(page, idx)]
        flush(doc, cur)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["vertex", "dataset", "chunk"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10, help="share of ROWS to put in validation")
    ap.add_argument("--gap", type=int, default=1,
                    help="pages closer than this share a cluster (1 = only consecutive pages; "
                         "use 2-3 to also separate near neighbours)")
    ap.add_argument("--block", type=int, default=20,
                    help="longest run of consecutive pages kept as one cluster; longer runs are cut into blocks")
    ap.add_argument("--holdout-docs", nargs="*", default=[], help="document names (substring match) held out entirely")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--gz", action="store_true", help="write .jsonl.gz")
    args = ap.parse_args()

    rows, keys, skipped = [], [], 0
    with opener(args.input, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            doc, page = doc_and_page(row, args.format)
            if doc is None:
                skipped += 1
                continue
            keys.append((doc, page, len(rows)))
            rows.append(row)
    print(f"read {len(rows)} rows ({skipped} skipped: unparseable or no document/page)")

    to_val = set()
    held_docs = {d for d, _, _ in keys if any(h.lower() in d.lower() for h in args.holdout_docs)}
    for doc, _, idx in keys:
        if doc in held_docs:
            to_val.add(idx)

    clusters = build_clusters([k for k in keys if k[0] not in held_docs], args.gap, args.block)
    rng = random.Random(args.seed)
    rng.shuffle(clusters)
    target = args.val_frac * len(rows)
    for doc, items in clusters:
        if len(to_val) >= target:
            break
        # prefer clusters that do not overshoot the target by a lot
        if len(to_val) + len(items) <= target * 1.25 or not to_val:
            to_val.update(idx for _, idx in items)

    # buffer: training pages within --gap pages of a validation page are not used for training
    val_pages = {(keys[i][0], keys[i][1]) for i in to_val}
    buffer = {i for i, (d, p, _) in enumerate(keys)
              if i not in to_val and any((d, p + k) in val_pages or (d, p - k) in val_pages
                                         for k in range(1, args.gap + 1))}

    ext = ".jsonl.gz" if args.gz else ".jsonl"
    os.makedirs(args.out, exist_ok=True)
    n_tr = n_va = n_buf = 0
    with opener(os.path.join(args.out, "train" + ext), "wt") as ftr, \
            opener(os.path.join(args.out, "val" + ext), "wt") as fva, \
            opener(os.path.join(args.out, "buffer" + ext), "wt") as fbu:
        for idx, row in enumerate(rows):
            line = json.dumps(row, ensure_ascii=False) + "\n"
            if idx in to_val:
                fva.write(line); n_va += 1
            elif idx in buffer:
                fbu.write(line); n_buf += 1
            else:
                ftr.write(line); n_tr += 1

    # ---- verification: no document/page-neighbour pair across splits (within --gap)
    side = {(d, p): ("val" if i in to_val else "buffer" if i in buffer else "train") for d, p, i in keys}
    leaks = [((d, p), (d, p + k)) for (d, p), v in side.items() for k in range(1, args.gap + 1)
             if (d, p + k) in side and {v, side[(d, p + k)]} == {"train", "val"}]

    def split_stats(name, idxs):
        ents, rels = [], []
        for i in idxs:
            g = graph_of(rows[i], args.format)
            if g:
                ents.append(len(g.get("entities") or []))
                rels.append(len(g.get("relationships") or []))
        docs = collections.Counter(keys[i][0] for i in idxs)
        out = {"rows": len(idxs), "documents": dict(docs)}
        if ents:
            out.update({"entities_per_page_mean": round(st.mean(ents), 1), "relationships_per_page_mean": round(st.mean(rels), 1)})
        return out

    tr_idx = [i for i in range(len(rows)) if i not in to_val and i not in buffer]
    report = {"args": vars(args), "train": split_stats("train", tr_idx), "val": split_stats("val", sorted(to_val)),
              "buffer_rows_not_used_for_training": len(buffer), "clusters_total": len(clusters), "held_out_documents": sorted(held_docs),
              "neighbour_leaks_across_splits": len(leaks)}
    with open(os.path.join(args.out, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {n_tr} train / {n_va} val rows ({100 * n_va / max(len(rows), 1):.1f}%) and set aside "
          f"{n_buf} buffer rows to {args.out}")
    if leaks:
        raise SystemExit(f"LEAK: {len(leaks)} neighbouring page pairs straddle the split, e.g. {leaks[:3]}")
    print("check passed: no page has a neighbour (within --gap) on the other side of the split")


if __name__ == "__main__":
    main()
