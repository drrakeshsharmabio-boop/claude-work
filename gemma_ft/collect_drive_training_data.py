"""collect_drive_training_data.py - harvest extra FBSFM training rows (and thinking) that already sit on Drive.

The PDF pipeline left one <book>_results.zip per processed PDF on Drive (Central_Library_Master/...).
Each zip holds the OCR of every page (metadata/page_NNN.json, document.md) and, once a KG extractor has
run, one graph per page (entities/pages/page_NNN.json). Extractors seen in the zips:

  notebooklm                 page JSON {ok, raw_entities, raw_relationships}; also training_data/ift_per_pdf.jsonl
  gemini_web_parallel_v3     page JSON {entities, relationships, _text_source}
  antigravity_parallel_v3    same, plus hard_pages.json (OCR quality flags per page)
  aistudio_web_parallel_v3   same, and for some pages the model's thought summary:
                             entities/thoughts/page_NNN_thought.md and "_thought" in the page JSON
                             (copied from the AI Studio web page, so it carries UI text such as
                             "Collapse to hide model thoughts" - removed here)

The AI Studio chats themselves are saved in "My Drive/Google AI Studio": every chat holds the full
production prompt, the thought parts (isThought) and the JSON answer, so it is a second, cleaner
source of thinking rows. Zips that only have OCR (no entities/pages) are counted, not used.

Output rows use the columns the v2 notebook already reads (section 3/6 of gemma_training_v2_promptfix):
  page, pdf_file, raw_ocr_text, extracted_graph (JSON string), plus thought ("" if none),
  data_source (extractor) and book_rel (<book>_results, as in the extraction chunks)
(not "source"/"book": the notebook's PDF_KEYS would read those as the PDF name of rows that have none.)
raw_ocr_text is the text the extractor saw: the document.md page section for the *_v3 extractors and
AI Studio, otherwise the OCR blocks of metadata/page_NNN.json in reading order.

Pages that are already in the current training set (--existing) are dropped: exact match of the
normalised text, or at least --dup-overlap of its sampled 6-word shingles seen in the training text.
Pages found twice on Drive (same book in two folders, AI Studio chat + zip of the same page) are kept
once, preferring the copy with a thought, then the one with more entities.

Outputs (--out, default gemma_ft/drive_data):
  drive_rows.jsonl.gz         new rows (not in --existing), one per page
  drive_thinking.jsonl.gz     the rows that have a thought
  combined_training.jsonl.gz  with --merge: existing rows + new rows in the same columns;
                              set DATASET to this file in the v2 notebook to train on both
  collect_report.json         counts per source, skip reasons, duplicates, books with most new pages
  harvest.jsonl.gz, done.txt  work files: a re-run skips sources listed in done.txt (delete both to
                              start over, or run with --finalize-only to rebuild the outputs)

Colab (Drive mounted at /content/drive):
  !python /content/drive/MyDrive/gemma_ft/collect_drive_training_data.py --limit 20     # quick test
  !python /content/drive/MyDrive/gemma_ft/collect_drive_training_data.py --merge        # everything
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
import zipfile
import zlib

DRIVE = "/content/drive/MyDrive"
PAGE_KEYS = ("page", "page_num", "page_number", "_page")
PDF_KEYS = ("pdf_file", "pdf_name", "source_pdf", "pdf", "source", "book")
FOCUS_RE = re.compile(r'Focus ONLY on PAGE (\d+) of source document "([^"]*)"')
PAGE_TEXT_RE = re.compile(r"=====PAGE TEXT=====\n(.*?)\n=====END PAGE TEXT=====", re.S)
DOC_PAGE_RE = re.compile(r"<!-- Page (\d+) -->")
PAGE_FILE_RE = re.compile(r"page_(\d+)\.json$")
THOUGHT_FILE_RE = re.compile(r"page_(\d+)_thought\.md$")
# lines of the AI Studio web page that end up in copied thoughts
UI_LINES = {"thoughts", "collapse to hide model thoughts", "expand to view model thoughts",
            "chevron_right", "expand_more", "expand_less"}
WORD_RE = re.compile("[0-9a-z%s-%s]+" % (chr(0x0900), chr(0x097F)))   # Latin digits/letters + Devanagari


def opener(path, mode):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


# --------------------------------------------------------------------------- text helpers
def words_of(text):
    return WORD_RE.findall(text.lower())


def text_key(text):
    return hashlib.blake2b(" ".join(words_of(text)).encode(), digest_size=8).hexdigest()


def shingles(words, n=6, mod=8):
    """Hashes of the n-word shingles whose hash is 0 mod `mod` (same sample on both sides, no alignment needed)."""
    out = set()
    for i in range(len(words) - n + 1):
        h = hash(tuple(words[i:i + n]))
        if h % mod == 0:
            out.add(h)
    return out


def clean_thought(text):
    lines = [l.rstrip() for l in (text or "").replace("\r", "").split("\n")]
    lines = [l for l in lines if l.strip().lower() not in UI_LINES]
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(t) < 40 or ('"entities"' in t and '"relationships"' in t):   # empty, or the answer itself
        return ""
    return t


def parse_answer(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|```$", "", t).strip()
    try:
        g = json.loads(t)
    except ValueError:
        return None
    return g if isinstance(g, dict) and isinstance(g.get("entities"), list) else None


def graph_json(entities, relationships):
    return json.dumps({"entities": entities, "relationships": relationships}, ensure_ascii=False)


# --------------------------------------------------------------------------- results zips
def zip_prefix(names):
    """Members are normally at the zip root; some zips wrap everything in one folder."""
    for n in names:
        i = n.find("metadata/page_")
        if i < 0:
            i = n.find("entities/pages/")
        if i >= 0:
            return n[:i]
    return ""


def doc_sections(doc):
    """{page: text} from document.md '<!-- Page N -->' markers (the text the *_v3 extractors were given)."""
    parts = DOC_PAGE_RE.split(doc)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[int(parts[i])] = out.get(int(parts[i]), "") + parts[i + 1]
    return {p: t.strip() for p, t in out.items()}


def blocks_text(meta):
    blocks = sorted(meta.get("blocks") or [], key=lambda b: b.get("reading_order") or 0)
    return "\n\n".join(b["text"].strip() for b in blocks if (b.get("text") or "").strip())


def rows_from_zip(path, book, stats):
    """Yield one candidate row per page that has a usable graph."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pre = zip_prefix(names)
        read = lambda n: z.read(pre + n).decode("utf-8", "replace")
        has = set(n[len(pre):] for n in names if n.startswith(pre))
        page_files = sorted(n for n in has if n.startswith("entities/pages/") and PAGE_FILE_RE.search(n))
        meta_files = [n for n in has if n.startswith("metadata/page_")]
        if not page_files:
            stats["zips_ocr_only"] += 1
            stats["pages_ocr_only"] += len(meta_files)
            return
        info = {}
        if "entities/all_pages_notebooklm.json" in has:
            try:
                info = json.loads(read("entities/all_pages_notebooklm.json"))
            except ValueError:
                pass
        src = info.get("extraction_source") or "unknown"
        stats[f"zips_with_graphs:{src}"] += 1
        thoughts = {}
        for n in has:
            m = THOUGHT_FILE_RE.search(n)
            if n.startswith("entities/thoughts/") and m:
                thoughts[int(m.group(1))] = read(n)
        hard = {}
        if "hard_pages.json" in has:
            try:
                hard = json.loads(read("hard_pages.json")).get("per_page") or {}
            except ValueError:
                pass
        sections = None
        for n in page_files:
            stats["pages_with_graph_file"] += 1
            try:
                d = json.loads(read(n))
            except ValueError:
                stats["skip:page_json_unreadable"] += 1
                continue
            page = d.get("page")
            if page is None:
                page = int(PAGE_FILE_RE.search(n).group(1))
            page = int(page)
            if d.get("ok") is False:
                stats["skip:extractor_failed"] += 1
                continue
            ents = d.get("entities", d.get("raw_entities")) or []
            rels = d.get("relationships", d.get("raw_relationships")) or []
            if not ents:
                stats["skip:empty_graph"] += 1
                continue
            if (hard.get(str(page)) or {}).get("has_repetition_loop"):
                stats["skip:ocr_repetition_loop"] += 1
                continue
            text = ""
            if d.get("_text_source") == "document.md":
                if sections is None:
                    sections = doc_sections(read("document.md")) if "document.md" in has else {}
                text = sections.get(page, "")
            if not text and f"metadata/page_{page:03d}.json" in has:
                try:
                    text = blocks_text(json.loads(read(f"metadata/page_{page:03d}.json")))
                except ValueError:
                    pass
            thought = clean_thought(d.get("_thought") or thoughts.get(page, ""))
            stats[f"pages_usable:{src}"] += 1
            stats[f"pages_with_thought:{src}"] += bool(thought)
            yield {"page": page, "pdf_file": d.get("pdf_file") or book + ".pdf", "raw_ocr_text": text,
                   "extracted_graph": graph_json(ents, rels), "thought": thought,
                   "data_source": "zip:" + src, "book_rel": book + "_results"}


# --------------------------------------------------------------------------- AI Studio chats
def rows_from_aistudio(path, stats):
    """Yield one row per (prompt, thought, answer) turn of a saved AI Studio chat."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, UnicodeDecodeError):
        stats["aistudio_files_not_json"] += 1
        return
    chunks = (d.get("chunkedPrompt") or {}).get("chunks") if isinstance(d, dict) else None
    if not chunks:
        stats["aistudio_files_not_chat"] += 1
        return
    stats["aistudio_chats"] += 1
    model = (d.get("runSettings") or {}).get("model", "")
    prompt, thought = None, []
    for c in chunks:
        if c.get("role") == "user":
            prompt, thought = c.get("text") or "", []
        elif c.get("isThought"):
            thought.append(c.get("text") or "")
        elif prompt is not None:
            stats["aistudio_turns"] += 1
            m, t = FOCUS_RE.search(prompt), PAGE_TEXT_RE.search(prompt)
            g = parse_answer(c.get("text"))
            if not (m and t):
                stats["skip:aistudio_not_kg_prompt"] += 1
            elif g is None:
                stats["skip:aistudio_no_valid_answer"] += 1   # errorMessage / cut-off answer
            elif not g["entities"]:
                stats["skip:empty_graph"] += 1
            else:
                th = clean_thought("\n\n".join(thought))
                stats["pages_usable:aistudio"] += 1
                stats["pages_with_thought:aistudio"] += bool(th)
                yield {"page": int(m.group(1)), "pdf_file": m.group(2), "raw_ocr_text": t.group(1),
                       "extracted_graph": graph_json(g["entities"], g.get("relationships") or []),
                       "thought": th, "data_source": "aistudio:" + model.replace("models/", ""),
                       "book_rel": os.path.splitext(m.group(2))[0] + "_results"}
            prompt, thought = None, []


# --------------------------------------------------------------------------- harvest (resumable)
def find_sources(args):
    zips, chats = [], []
    for root in args.roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if os.path.abspath(os.path.join(dirpath, x)) != os.path.abspath(args.out)]
            zips.extend(os.path.join(dirpath, f) for f in filenames if f.endswith("_results.zip"))
    if args.aistudio_dir and os.path.isdir(args.aistudio_dir):
        chats = [os.path.join(args.aistudio_dir, f) for f in os.listdir(args.aistudio_dir)
                 if os.path.isfile(os.path.join(args.aistudio_dir, f))]
    return sorted(zips), sorted(chats)


def harvest(args):
    """Append rows of every source not yet in done.txt to harvest.jsonl.gz; returns the counts of all runs."""
    harvest_path, done_path = os.path.join(args.out, "harvest.jsonl.gz"), os.path.join(args.out, "done.txt")
    stats_path = os.path.join(args.out, "harvest_stats.json")
    done = set()
    if os.path.exists(done_path):
        # one JSON string per line: AI Studio file names are the chat titles and contain newlines
        with open(done_path, encoding="utf-8") as f:
            done = {json.loads(l) for l in f if l.strip()}
    t0 = time.time()
    zips, chats = find_sources(args)
    print(f"found {len(zips)} *_results.zip under {args.roots} and {len(chats)} files in {args.aistudio_dir} "
          f"({time.time() - t0:.0f}s); {len(done)} already done")
    todo = [("zip", p) for p in zips] + [("aistudio", p) for p in chats]
    todo = [(k, p) for k, p in todo if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    run = collections.Counter()
    for i, (kind, path) in enumerate(todo, 1):
        rows = []
        try:
            if kind == "zip":
                if args.max_zip_mb and os.path.getsize(path) > args.max_zip_mb * 1e6:
                    run["skip:zip_too_big"] += 1
                    continue
                book = os.path.basename(path)[: -len("_results.zip")]
                rows = list(rows_from_zip(path, book, run))
            else:
                rows = list(rows_from_aistudio(path, run))
        except (zipfile.BadZipFile, OSError, KeyError) as e:
            run["skip:unreadable_source"] += 1
            print(f"  ! {path}: {type(e).__name__} {e}")
            continue
        # one gzip member per source: append-safe, and readable as one stream afterwards
        with gzip.open(harvest_path, "at", encoding="utf-8") as f:
            for r in rows:
                r["origin"] = os.path.relpath(path, DRIVE) if path.startswith(DRIVE) else path
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(path, ensure_ascii=False) + "\n")
        run["rows_harvested"] += len(rows)
        run[f"sources_done:{kind}"] += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} sources, {run['rows_harvested']} rows, {time.time() - t0:.0f}s", flush=True)
            save_stats(stats_path, run)
            run = collections.Counter()
    save_stats(stats_path, run)
    return load_stats(stats_path)


def load_stats(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return collections.Counter(json.load(f))
    return collections.Counter()


def save_stats(path, run):
    total = load_stats(path)
    total.update(run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(total), f, indent=1)


# --------------------------------------------------------------------------- finalize
def row_page(row):
    for k in PAGE_KEYS:
        if row.get(k) not in (None, ""):
            return int(row[k])
    try:
        g = json.loads(row["extracted_graph"])
    except (KeyError, TypeError, ValueError):
        return None
    if isinstance(g, dict):
        if g.get("page") not in (None, ""):
            return int(g["page"])
        for e in g.get("entities") or []:
            if isinstance(e, dict) and e.get("_page", e.get("page")) is not None:
                return int(e.get("_page", e.get("page")))
    return None


def row_pdf(row):
    for k in PDF_KEYS:
        if row.get(k):
            return os.path.basename(str(row[k]))
    return ""


def index_existing(args, combined):
    """Fingerprints of the training text already used; with --merge also copies those rows to `combined`."""
    keys, shing, n = set(), set(), 0
    for path in args.existing:
        if not os.path.exists(path):
            print(f"  (existing dataset not found, skipped: {path})")
            continue
        with opener(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("raw_ocr_text") or ""
                keys.add(text_key(text))
                shing |= shingles(words_of(text), mod=args.shingle_mod)
                n += 1
                if combined is not None:
                    combined.write(json.dumps({
                        "page": row_page(row), "pdf_file": row_pdf(row), "raw_ocr_text": text,
                        "extracted_graph": row.get("extracted_graph"), "thought": row.get("thought") or "",
                        "data_source": "existing", "book_rel": ""}, ensure_ascii=False) + "\n")
    print(f"indexed {n} existing training rows ({len(shing)} sampled shingles)")
    return keys, shing, n


def iter_harvest(path):
    """Rows of harvest.jsonl.gz; stops cleanly at a tail cut off by an interrupted run."""
    if not os.path.exists(path):
        return
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass
        except (EOFError, OSError, zlib.error) as e:
            print(f"  (harvest file ends with an incomplete write, ignored: {type(e).__name__})")


def finalize(args, harvest_path, harvest_stats):
    out = lambda name: os.path.join(args.out, name)
    stats = collections.Counter()
    combined = gzip.open(out("combined_training.jsonl.gz"), "wt", encoding="utf-8") if args.merge else None
    keys, shing, n_existing = index_existing(args, combined)

    # pass 1: choose one copy per page text, mark pages already trained on
    best, seen_in_training = {}, set()
    n_rows = 0
    for i, r in enumerate(iter_harvest(harvest_path)):
        n_rows += 1
        text = r["raw_ocr_text"]
        if len(text.strip()) < args.min_chars:
            stats["skip:text_too_short"] += 1
            continue
        k = text_key(text)
        if k in keys:
            seen_in_training.add(k)
        elif shing:
            s = shingles(words_of(text), mod=args.shingle_mod)
            if len(s) >= 3 and len(s & shing) >= args.dup_overlap * len(s):
                seen_in_training.add(k)
        score = (bool(r["thought"]), r["data_source"].startswith("aistudio"),
                 len(json.loads(r["extracted_graph"])["entities"]))
        if k not in best or score > best[k][0]:
            best[k] = (score, i)
    keep = {i for k, (_, i) in best.items() if k not in seen_in_training}
    stats["harvested_rows_total"] = n_rows
    stats["duplicate_copies_on_drive"] = n_rows - stats["skip:text_too_short"] - len(best)
    stats["already_in_training_set"] = len(seen_in_training)

    # pass 2: write the chosen rows
    per_source, per_book = collections.Counter(), collections.Counter()
    th_src, ents, rels = collections.Counter(), [], []
    n_new = n_think = 0
    with gzip.open(out("drive_rows.jsonl.gz"), "wt", encoding="utf-8") as fr, \
            gzip.open(out("drive_thinking.jsonl.gz"), "wt", encoding="utf-8") as ft:
        for i, r in enumerate(iter_harvest(harvest_path)):
            if i not in keep:
                continue
            r.pop("origin", None)
            line = json.dumps(r, ensure_ascii=False) + "\n"
            fr.write(line)
            if combined is not None:
                combined.write(line)
            n_new += 1
            per_source[r["data_source"]] += 1
            per_book[r["book_rel"]] += 1
            g = json.loads(r["extracted_graph"])
            ents.append(len(g["entities"]))
            rels.append(len(g.get("relationships") or []))
            if r["thought"]:
                ft.write(line)
                n_think += 1
                th_src[r["data_source"]] += 1
    if combined is not None:
        combined.close()

    report = {"args": {k: v for k, v in vars(args).items()}, "existing_rows": n_existing,
              "new_rows": n_new, "new_rows_with_thought": n_think,
              "new_rows_by_source": dict(per_source.most_common()),
              "thought_rows_by_source": dict(th_src.most_common()),
              "entities_per_page_mean": round(st.mean(ents), 1) if ents else None,
              "relationships_per_page_mean": round(st.mean(rels), 1) if rels else None,
              "books_with_new_rows": len(per_book), "top_books": dict(per_book.most_common(25)),
              "harvest_counts": dict(sorted(harvest_stats.items())), "finalize_counts": dict(sorted(stats.items()))}
    with open(out("collect_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in report if k not in ("args", "top_books")}, indent=2, ensure_ascii=False))
    print(f"\nnew rows: {n_new} ({n_think} with a thought) -> {out('drive_rows.jsonl.gz')}")
    if combined is not None:
        print(f"combined ({n_existing} existing + {n_new} new) -> {out('combined_training.jsonl.gz')}\n"
              f"   in the v2 notebook set  DATASET = \"{out('combined_training.jsonl.gz')}\"")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[f"{DRIVE}/Central_Library_Master"],
                    help="folders searched (recursively) for *_results.zip")
    ap.add_argument("--aistudio-dir", default=f"{DRIVE}/Google AI Studio", help="saved AI Studio chats ('' to skip)")
    ap.add_argument("--existing", nargs="*", default=[f"{DRIVE}/gemma_ft/gemma_training_dataset.jsonl.gz"],
                    help="current training rows; pages already in them are not added again")
    ap.add_argument("--out", default=f"{DRIVE}/gemma_ft/drive_data")
    ap.add_argument("--merge", action="store_true", help="also write combined_training.jsonl.gz (existing + new)")
    ap.add_argument("--limit", type=int, default=0, help="process only this many new sources (for a test run)")
    ap.add_argument("--max-zip-mb", type=float, default=0, help="skip zips larger than this (0 = no limit)")
    ap.add_argument("--min-chars", type=int, default=150, help="drop pages with less OCR text than this")
    ap.add_argument("--dup-overlap", type=float, default=0.5,
                    help="share of a page's sampled shingles found in the training text that makes it a duplicate")
    ap.add_argument("--shingle-mod", type=int, default=8, help="keep 1 in N shingles for the duplicate check")
    ap.add_argument("--finalize-only", action="store_true", help="skip the harvest, rebuild outputs from harvest.jsonl.gz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    harvest_path = os.path.join(args.out, "harvest.jsonl.gz")
    if args.finalize_only:
        harvest_stats = load_stats(os.path.join(args.out, "harvest_stats.json"))
    else:
        harvest_stats = harvest(args)
    finalize(args, harvest_path, harvest_stats)


if __name__ == "__main__":
    sys.exit(main())
