"""validate_gemma_lora.py - validate the fine-tuned Gemma 4 31B FBSFM LoRA adapter.

Run in a FRESH Colab runtime (Runtime > Disconnect and delete runtime, then reconnect), so
no other model is holding GPU memory:

    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
    from google.colab import drive; drive.mount("/content/drive")

    # A) pages from the training file (ground truth available -> P/R/F1)
    !python /content/drive/MyDrive/gemma_ft/validate_gemma_lora.py --data train --n 20

    # B) UNSEEN pages from other_resources_in (no ground truth -> grounding checks only)
    !python /content/drive/MyDrive/gemma_ft/validate_gemma_lora.py --data chunk --n 20

Prompt style (--prompt, default auto = read <adapter>/prompt_style.txt, else v1):
  v1  the format the FIRST adapter (gemma_fbsfm_lora) was trained on: short "Below is an OCR
      page..." prompt, generation prompt WITHOUT Gemma 4's `<|channel>thought\n<channel|>`
      suffix, reply starts with "### Extracted JSON:".
  v2  the format of gemma_training_v2_promptfix.ipynb: the full production Ed-Machine prompt
      and the chat template's own generation prompt (with the thought-channel suffix), reply is
      bare JSON. With --data train, v2 needs rows that carry a `prompt` field (the held-out
      file written by the v2 notebook does).

What it fixes compared to the notebook cells:
  * prompts the model in the SAME format it was trained on (see --prompt);
  * stops on `<turn|>` as well as <eos>;
  * loads the adapter with text_only=True (without it every LoRA key is "missing" and the
    adapter is silently NOT applied);
  * strips Gemma 4's empty thinking block `<|channel>thought\\n<channel|>` before parsing;
  * allows 4096 new tokens (2048 truncated the JSON before "relationships") and uses a mild
    repetition penalty (1.2 damages JSON);
  * salvages truncated JSON and reports it separately instead of calling it a parse failure;
  * scores the output: schema, dangling relationship endpoints, positional names, Romanized
    name_hi, entities that are not grounded in the page text, and P/R/F1 vs ground truth.

NOTE: the original training file has no held-out split, so --data train on it samples TRAINING
rows: it tells you whether the format was learned, not whether it generalises. Use --data chunk,
or the held-out file written by the v2 notebook, for that.
"""
import argparse
import glob
import gzip
import json
import os
import random
import re
import time

VALID_CLASSES = {"S", "B", "F", "P", "C", "M"}
BAD_PREFIXES = ("FIRST_", "SECOND_", "THIRD_", "THIS_", "THAT_", "TARGET_", "R1_", "R2_")
SNAKE_RE = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
STOP = {"with", "from", "type", "that", "this", "into", "their", "page", "figure", "table"}
CLEAN_TOKENS = ("<turn|>", "<|turn>", "<eos>", "<pad>", "<bos>", "<end_of_turn>")
THOUGHT_SUFFIX = "<|channel>thought\n<channel|>"  # Gemma 4 template adds this when thinking is off

# The exact user prompt used during training (see the training cell of the notebook).
TRAIN_PROMPT = (
    "Below is an OCR page from a biology textbook. Extract the knowledge graph in JSON format."
    "\n\n### Page Text:\n{page_text}"
)


# --------------------------------------------------------------------------- parsing
def clean_generation(text):
    """Remove Gemma 4's thinking-channel prefix, turn markers and the training reply header."""
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    for tok in CLEAN_TOKENS:
        text = text.replace(tok, "")
    text = text.strip()
    if "### Extracted JSON:" in text:
        text = text.split("### Extracted JSON:", 1)[1].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return text


def _raw_decode_items(text, key):
    """Collect every COMPLETE object of the array `"key": [ ... ]`, even if the text is cut off."""
    m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), text)
    if not m:
        return []
    dec, pos, items = json.JSONDecoder(), m.end(), []
    while True:
        while pos < len(text) and text[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            obj, pos = dec.raw_decode(text, pos)
        except ValueError:
            break  # cut off mid-object: drop the partial one
        items.append(obj)
    return items


def parse_generation(raw):
    """Return (graph_dict_or_None, status) where status in ok | salvaged | failed."""
    text = clean_generation(raw)
    for cand in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and isinstance(d.get("entities"), list):
                d.setdefault("relationships", [])
                return d, "ok"
        except ValueError:
            pass
    ents = _raw_decode_items(text, "entities")
    if ents:
        return {"entities": ents, "relationships": _raw_decode_items(text, "relationships")}, "salvaged"
    return None, "failed"


# --------------------------------------------------------------------------- scoring
def _stems(s):
    # short words use a 4-char stem so "ovary" still matches "ovaries"
    return {t[:4] if len(t) <= 6 else t[:5] for t in re.findall(r"[a-z]{4,}", s.lower()) if t not in STOP}


def is_english_page(page_text):
    letters = re.findall(r"[A-Za-z\u0900-\u097F]", page_text)
    if not letters:
        return False
    ascii_letters = sum(1 for ch in letters if ch.isascii())
    return ascii_letters / len(letters) > 0.7


def score_graph(graph, page_text):
    ents = [e for e in graph.get("entities", []) if isinstance(e, dict)]
    rels = [r for r in graph.get("relationships", []) if isinstance(r, dict)]
    names = [str(e.get("name", "")) for e in ents]
    nameset = set(names)
    page_l = page_text.lower()
    check_grounding = is_english_page(page_text)

    issues = {"bad_name_format": [], "positional_name": [], "bad_class": [], "romanized_hi": [],
              "ungrounded": [], "duplicate_name": [], "dangling_rel": []}
    seen = set()
    for e in ents:
        n = str(e.get("name", ""))
        if not SNAKE_RE.match(n):
            issues["bad_name_format"].append(n)
        if n.startswith(BAD_PREFIXES):
            issues["positional_name"].append(n)
        if e.get("fbsfm_class") not in VALID_CLASSES:
            issues["bad_class"].append(n)
        hi = str(e.get("name_hi", ""))
        if hi and not DEVANAGARI_RE.search(hi):
            issues["romanized_hi"].append(n)
        if n in seen:
            issues["duplicate_name"].append(n)
        seen.add(n)
        if check_grounding:
            st = _stems(str(e.get("name_en", "")) or n.replace("_", " "))
            if st and sum(1 for s in st if s in page_l) / len(st) < 0.5:
                issues["ungrounded"].append(n)
    for r in rels:
        if r.get("source") not in nameset or r.get("target") not in nameset:
            issues["dangling_rel"].append("%s -%s-> %s" % (r.get("source"), r.get("type"), r.get("target")))

    return {
        "n_entities": len(ents),
        "n_relationships": len(rels),
        "grounding_checked": check_grounding,
        "ungrounded_frac": (len(issues["ungrounded"]) / len(ents)) if (ents and check_grounding) else None,
        "dangling_frac": (len(issues["dangling_rel"]) / len(rels)) if rels else None,
        "issues": issues,
    }


def compare_to_truth(graph, truth):
    """Exact-match P/R/F1 on entity names and on (source, type, target) triples."""
    def prf(pred, gold):
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if (p + r) else 0.0}

    pe = {e.get("name") for e in graph.get("entities", []) if isinstance(e, dict)}
    ge = {e.get("name") for e in truth.get("entities", []) if isinstance(e, dict)}
    pr = {(r.get("source"), r.get("type"), r.get("target")) for r in graph.get("relationships", []) if isinstance(r, dict)}
    gr = {(r.get("source"), r.get("type"), r.get("target")) for r in truth.get("relationships", []) if isinstance(r, dict)}
    return {"entities": prf(pe, ge), "relationships": prf(pr, gr)}


# --------------------------------------------------------------------------- data
def sample_train_rows(path, n, seed):
    """Reservoir-sample n rows from the (possibly huge) .jsonl.gz without loading it all."""
    rng, res, seen = random.Random(seed), [], 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seen += 1
            if len(res) < n:
                res.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    res[j] = row
    return res, seen


def page_text_from_prompt(prompt):
    m = re.search(r"=====PAGE TEXT=====\s*(.*?)\s*=====END PAGE TEXT=====", prompt, re.S)
    return m.group(1) if m else prompt


def resolve_prompt_style(args):
    if args.prompt != "auto":
        return args.prompt
    marker = os.path.join(args.adapter, "prompt_style.txt")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            return f.read().strip() or "v1"
    return "v1"


def build_cases(args):
    cases = []
    if args.data == "train":
        rows, total = sample_train_rows(args.dataset, args.n, args.seed)
        print("Sampled %d of %d rows from %s." % (len(rows), total, args.dataset))
        for r in rows:
            try:
                truth = json.loads(r["extracted_graph"])
            except Exception:
                truth = None
            if args.prompt == "v2":
                if "prompt" not in r:
                    raise SystemExit("--prompt v2 with --data train needs rows with a `prompt` field; use the "
                                     "held-out file written by the v2 notebook as --dataset.")
                prompt = r["prompt"]
            else:
                prompt = TRAIN_PROMPT.format(page_text=r["raw_ocr_text"])
            cases.append({"id": str(r.get("page", "?")), "page_text": r["raw_ocr_text"],
                          "prompt": prompt, "truth": truth})
    else:
        files = sorted(glob.glob(os.path.join(args.chunks, "other_resources_chunk_*.jsonl.gz")))
        if not files:
            raise SystemExit("No other_resources_chunk_*.jsonl.gz in %s" % args.chunks)
        rows = []
        with gzip.open(files[0], "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        random.Random(args.seed).shuffle(rows)
        print("Using %d of %d pages from %s (UNSEEN pages, no ground truth)." % (min(args.n, len(rows)), len(rows), files[0]))
        for r in rows[: args.n]:
            page_text = page_text_from_prompt(r["prompt"])
            prompt = r["prompt"] if args.prompt == "v2" else TRAIN_PROMPT.format(page_text=page_text)
            cases.append({"id": "%s p%s" % (r.get("pdf_name", "?"), r.get("page", "?")),
                          "page_text": page_text, "prompt": prompt, "truth": None})
    return cases


# --------------------------------------------------------------------------- model
def load_model(args):
    import torch  # noqa: F401  (imported before unsloth on purpose)
    from unsloth import FastLanguageModel

    print("Loading adapter %s (text_only=True) ..." % args.adapter)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=args.max_seq, dtype=None,
        load_in_4bit=True, text_only=True,
    )
    FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def chat_prompt(tok, prompt, style):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if style == "v1" and text.endswith(THOUGHT_SUFFIX):
        text = text[: -len(THOUGHT_SUFFIX)]  # v1 was trained on "<|turn>model\n" + reply, no thought channel
    return text


def stop_ids(tok):
    ids = [tok.eos_token_id]
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    if isinstance(turn_end, int) and turn_end != tok.unk_token_id:
        ids.append(turn_end)
    return [i for i in ids if i is not None]


def generate(model, tok, prompts, args):
    import torch
    chat = [chat_prompt(tok, p, args.prompt) for p in prompts]
    enc = tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                             repetition_penalty=args.rep_penalty, pad_token_id=tok.pad_token_id,
                             eos_token_id=stop_ids(tok))
    res = []
    for j in range(len(prompts)):
        gen = out[j][enc["input_ids"].shape[1]:]
        n_tok = int((gen != tok.pad_token_id).sum())
        res.append((tok.decode(gen, skip_special_tokens=False), n_tok))
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["train", "chunk"], default="train")
    ap.add_argument("--prompt", choices=["auto", "v1", "v2"], default="auto")
    ap.add_argument("--adapter", default="/content/drive/MyDrive/gemma_ft/gemma_fbsfm_lora")
    ap.add_argument("--dataset", default="/content/drive/MyDrive/gemma_ft/gemma_training_dataset.jsonl.gz")
    ap.add_argument("--chunks", default="/content/drive/MyDrive/gemma_ft/other_resources_in")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--rep-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--report", default="/content/drive/MyDrive/gemma_ft/validation_report.json")
    args = ap.parse_args()
    args.prompt = resolve_prompt_style(args)
    print("Prompt style: %s" % args.prompt)

    cases = build_cases(args)
    model, tok = load_model(args)

    results = []
    for k in range(0, len(cases), args.batch):
        part = cases[k:k + args.batch]
        t0 = time.time()
        outs = generate(model, tok, [c["prompt"] for c in part], args)
        dt = (time.time() - t0) / len(part)
        for c, (raw, n_tok) in zip(part, outs):
            graph, status = parse_generation(raw)
            truncated = n_tok >= args.max_new
            rec = {"id": c["id"], "status": status, "truncated": truncated, "gen_tokens": n_tok,
                   "sec": round(dt, 1), "raw_head": raw[:300]}
            if graph is not None:
                rec["score"] = score_graph(graph, c["page_text"])
                if c["truth"]:
                    rec["vs_truth"] = compare_to_truth(graph, c["truth"])
            results.append(rec)
            s = rec.get("score", {})
            print("[%-28s] %-8s trunc=%-5s ents=%-3s rels=%-3s ungrounded=%s dangling=%s" % (
                c["id"][:28], status, truncated, s.get("n_entities", "-"), s.get("n_relationships", "-"),
                "n/a" if s.get("ungrounded_frac") is None else "%.0f%%" % (100 * s["ungrounded_frac"]),
                "n/a" if s.get("dangling_frac") is None else "%.0f%%" % (100 * s["dangling_frac"])))

    # ---- aggregate
    n = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    salv = sum(1 for r in results if r["status"] == "salvaged")
    fail = sum(1 for r in results if r["status"] == "failed")
    trunc = sum(1 for r in results if r["truncated"])
    scored = [r["score"] for r in results if "score" in r]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    ung = mean([s["ungrounded_frac"] for s in scored])
    dang = mean([s["dangling_frac"] for s in scored])
    pos = sum(len(s["issues"]["positional_name"]) for s in scored)
    rom = sum(len(s["issues"]["romanized_hi"]) for s in scored)
    badc = sum(len(s["issues"]["bad_class"]) for s in scored)
    truth = [r["vs_truth"] for r in results if "vs_truth" in r]

    print("\n================ SUMMARY (%d pages, data=%s, prompt=%s) ================" % (n, args.data, args.prompt))
    print("valid JSON        : %d ok, %d salvaged (truncated), %d failed" % (ok, salv, fail))
    print("hit max_new_tokens: %d / %d" % (trunc, n))
    print("avg entities/page : %.1f   avg relationships/page: %.1f" % (
        mean([s["n_entities"] for s in scored]) or 0, mean([s["n_relationships"] for s in scored]) or 0))
    print("ungrounded ents   : %s   (not found in page text; English pages only)" % ("n/a" if ung is None else "%.1f%%" % (100 * ung)))
    print("dangling rels     : %s   (source/target not in entities)" % ("n/a" if dang is None else "%.1f%%" % (100 * dang)))
    print("positional names=%d  romanized name_hi=%d  bad fbsfm_class=%d" % (pos, rom, badc))
    if truth:
        for key in ("entities", "relationships"):
            print("vs ground truth %-13s P=%.2f R=%.2f F1=%.2f" % (
                key, mean([t[key]["precision"] for t in truth]), mean([t[key]["recall"] for t in truth]),
                mean([t[key]["f1"] for t in truth])))

    # Heuristic gates - tune to taste.
    gates = [("valid-or-salvaged JSON >= 95%", n and (ok + salv) / n >= 0.95),
             ("full valid JSON >= 90%", n and ok / n >= 0.90),
             ("ungrounded entities <= 5%", ung is None or ung <= 0.05),
             ("dangling relationships <= 5%", dang is None or dang <= 0.05),
             ("no positional names / romanized name_hi", pos == 0 and rom == 0)]
    print("\nHeuristic gates:")
    for name, passed in gates:
        print("  [%s] %s" % ("PASS" if passed else "FAIL", name))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    print("\nFull per-page report (incl. ungrounded entity names): %s" % args.report)


if __name__ == "__main__":
    main()
