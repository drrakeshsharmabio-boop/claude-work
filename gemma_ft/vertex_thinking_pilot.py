"""vertex_thinking_pilot.py - re-run Gemini extractions WITH readable thought summaries, then check them.

Your existing Vertex batch output (predictions.jsonl) has the answer text plus an encrypted
`thoughtSignature` only, because the requests did not ask for thought summaries. This script:

  prepare   builds a new batch-input JSONL from the requests in an existing predictions file (or from
            val.jsonl / train.jsonl written by make_train_val_split.py), identical prompt, schema and
            thinkingLevel, plus  thinkingConfig.includeThoughts = true. Prints the token volume of the
            original run so you can price the job with your own rates.
  submit    (optional) uploads the input to Cloud Storage and starts the Vertex AI batch job with the
            google-genai SDK, then waits for it. If anything here errors, use the console route below.
  analyze   reads the batch output and answers the questions that decide whether thinking rows are
            usable: did the batch return READABLE thought parts? how long are they? is the answer still
            valid JSON? how different is it from the first run? Can also write training rows.

Console route (no SDK needed): upload pilot_input.jsonl to a Cloud Storage bucket, then Vertex AI ->
Batch inference -> create job with the same Gemini model, input = that file, output = a bucket folder.
Download the result file(s) and run `analyze`.

Suggested order (cheapest first):
  python vertex_thinking_pilot.py prepare --input split_v1/val.jsonl   --out pilot_val_input.jsonl      # 30 pages
  ... run the job, then:
  python vertex_thinking_pilot.py analyze --output <results file or folder> --original predictions.jsonl
  Only if the summaries look usable: repeat with split_v1/train.jsonl (270 pages), and later a larger set.

NOTE: the `submit` step uses the google-genai SDK from memory of its API and could not be run where this
file was written (no Google credentials). Official Vertex docs were not reachable either, so whether batch
jobs return thought summaries is exactly what `analyze` tells you.
"""
import argparse
import collections
import copy
import glob
import json
import os
import re
import statistics as st
import sys
import time

FOCUS_RE = re.compile(r'Focus ONLY on PAGE (\d+) of source document "([^"]+)"')


# --------------------------------------------------------------------------- helpers
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def prompt_of(row):
    return "".join(p.get("text", "") for p in row["request"]["contents"][0]["parts"])


def key_of(row):
    m = FOCUS_RE.search(prompt_of(row))
    return (m.group(2), int(m.group(1))) if m else None


def split_parts(row):
    """Return (thought_text, answer_text, n_signature_parts) from a batch response row."""
    try:
        parts = row["response"]["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return "", "", 0
    thought = "\n".join(p.get("text", "") for p in parts if p.get("thought") and p.get("text"))
    answer = "".join(p.get("text", "") for p in parts if not p.get("thought") and p.get("text"))
    sigs = sum(1 for p in parts if p.get("thoughtSignature"))
    return thought, answer, sigs


def usage(row):
    u = (row.get("response") or {}).get("usageMetadata") or {}
    return (int(u.get("promptTokenCount", 0) or 0), int(u.get("thoughtsTokenCount", 0) or 0),
            int(u.get("candidatesTokenCount", 0) or 0))


def collect_outputs(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
        if not files:
            raise SystemExit(f"no .jsonl files under {path}")
    else:
        files = [path]
    rows = []
    for fpath in files:
        rows.extend(read_jsonl(fpath))
    return rows, files


# --------------------------------------------------------------------------- prepare
def cmd_prepare(args):
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    n = 0
    p_tok = t_tok = c_tok = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            req = copy.deepcopy(row["request"])
            gc = req.setdefault("generationConfig", {})
            tc = gc.setdefault("thinkingConfig", {})
            if args.level:
                tc["thinkingLevel"] = args.level
            tc["includeThoughts"] = True  # <- the only intended change
            f.write(json.dumps({"request": req}, ensure_ascii=False) + "\n")
            n += 1
            if "response" in row:
                a, b, c = usage(row)
                p_tok += a; t_tok += b; c_tok += c
    print(f"wrote {n} requests to {args.out}  (thinkingConfig = {tc})")
    if p_tok:
        print("token volume of the ORIGINAL run for these pages (price with your own rates):")
        print(f"  prompt {p_tok:,}   thinking {t_tok:,}   answer {c_tok:,}")
        print("  The new run needs about the same, plus a small amount for the returned thought summaries.")
    print("next: submit with the console, or  python vertex_thinking_pilot.py submit --input", args.out, "...")


# --------------------------------------------------------------------------- submit
def cmd_submit(args):
    try:
        from google import genai
        from google.genai import types
        from google.cloud import storage
    except ImportError:
        raise SystemExit("pip install google-genai google-cloud-storage   (and authenticate: gcloud auth application-default login)")
    bucket_name, _, prefix = args.gcs_prefix.replace("gs://", "").partition("/")
    prefix = prefix.rstrip("/")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    blob_name = f"{prefix}/{stamp}/input.jsonl".lstrip("/")
    storage.Client(project=args.project).bucket(bucket_name).blob(blob_name).upload_from_filename(args.input)
    src = f"gs://{bucket_name}/{blob_name}"
    dest = f"gs://{bucket_name}/{prefix}/{stamp}/output/".replace("//", "/").replace("gs:/", "gs://")
    print("uploaded", src)

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    job = client.batches.create(model=args.model, src=src,
                                config=types.CreateBatchJobConfig(dest=dest, display_name=f"thinking-pilot-{stamp}"))
    print("job:", job.name, "state:", job.state)
    done = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED")
    while not any(d in str(job.state) for d in done):
        time.sleep(args.poll)
        job = client.batches.get(name=job.name)
        print(time.strftime("%H:%M:%S"), job.state, flush=True)
    print("finished:", job.state, "\noutput folder:", dest,
          "\n-> download it (gsutil -m cp -r) and run `analyze --output <folder>`")


# --------------------------------------------------------------------------- analyze
def entity_names(answer):
    try:
        g = json.loads(answer)
        return {e.get("name") for e in g.get("entities", []) if isinstance(e, dict)}, len(g.get("relationships", []))
    except Exception:
        return None, None


def cmd_analyze(args):
    rows, files = collect_outputs(args.output)
    print(f"read {len(rows)} rows from {len(files)} file(s)")
    original = {}
    if args.original:
        for r in read_jsonl(args.original):
            k = key_of(r)
            if k:
                original[k] = r

    stat = collections.Counter()
    th_chars, th_tok_api, jacc, relcmp, train_rows = [], [], [], [], []
    examples = []
    for row in rows:
        if row.get("status"):
            stat["row_error"] += 1
            continue
        thought, answer, sigs = split_parts(row)
        stat["rows"] += 1
        stat["with_signature"] += bool(sigs)
        if thought.strip():
            stat["with_readable_thought"] += 1
            th_chars.append(len(thought))
        names, nrel = entity_names(answer)
        if names is None:
            stat["answer_not_valid_json"] += 1
            continue
        stat["answer_valid_json"] += 1
        if thought.strip() and ('"entities"' in thought and '"relationships"' in thought):
            stat["thought_contains_the_json"] += 1
        th_tok_api.append(usage(row)[1])
        k = key_of(row)
        if k in original:
            o_names, o_rel = entity_names(split_parts(original[k])[1])
            if o_names:
                jacc.append(len(names & o_names) / max(len(names | o_names), 1))
                relcmp.append((o_rel, nrel))
        if thought.strip():
            train_rows.append({"document": k[0] if k else None, "page": k[1] if k else None,
                               "prompt": prompt_of(row), "thought": thought.strip(), "answer": answer.strip()})
            if len(examples) < 2:
                examples.append((k, thought))

    print("\n================ SUMMARY ================")
    print(f"rows ok                       : {stat['rows']}  (errors: {stat['row_error']})")
    print(f"answer is valid JSON          : {stat['answer_valid_json']} / {stat['rows']}")
    print(f"rows with READABLE thought    : {stat['with_readable_thought']} / {stat['rows']}"
          f"   <- if 0, the batch API did not return summaries; use non-batch calls or another route")
    print(f"rows that still carry a signature: {stat['with_signature']}")
    if th_chars:
        qs = sorted(th_chars)
        print(f"thought summary length (chars): median {int(st.median(th_chars))}, p90 {qs[int(.9 * len(qs))]}, max {qs[-1]}"
              f"   (~{int(st.median(th_chars)) // 4} tokens median)")
        print(f"thought repeats the answer JSON: {stat['thought_contains_the_json']} rows (should be ~0)")
    if th_tok_api:
        print(f"billed thinking tokens (API)   : median {int(st.median(th_tok_api))}  -> summaries are only a small part of this")
    if jacc:
        print(f"vs original run: entity-name overlap (Jaccard) mean {st.mean(jacc):.2f}; relationships original "
              f"{st.mean(a for a, _ in relcmp):.1f} vs new {st.mean(b for _, b in relcmp):.1f} per page")
        print("   (two runs of the same prompt differ somewhat; this sets the ceiling for 'agreement with ground truth')")
    for k, th in examples:
        print(f"\n--- example thought summary {k} ({len(th)} chars) ---\n{th[:900]}{' ...' if len(th) > 900 else ''}")
    if args.emit_train and train_rows:
        with open(args.emit_train, "w", encoding="utf-8") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(train_rows)} thinking rows (prompt / thought / answer) to {args.emit_train}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True, help="Vertex predictions-style JSONL (rows with a `request`)")
    p.add_argument("--out", default="pilot_input.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--level", default="", help="override thinkingLevel (default: keep the original, e.g. HIGH)")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("submit")
    p.add_argument("--input", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--location", required=True, help="Vertex region for the batch job")
    p.add_argument("--model", default="gemini-3.8-flash", help="same model id as the first run")
    p.add_argument("--gcs-prefix", required=True, help="gs://bucket/folder for input and output")
    p.add_argument("--poll", type=int, default=60)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("analyze")
    p.add_argument("--output", required=True, help="batch result .jsonl or a folder containing it")
    p.add_argument("--original", default="", help="the first run's predictions.jsonl, to compare answers")
    p.add_argument("--emit-train", default="", help="write prompt/thought/answer rows for training")
    p.set_defaults(fn=cmd_analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
