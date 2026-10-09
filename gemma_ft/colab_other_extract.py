"""colab_other_extract.py - Colab GPU extractor for OTHER RESOURCES using fine-tuned Gemma LoRA.

v4 changes vs v3:
  * loads the 31B 4-bit base + adapter through Unsloth with text_only=True (the adapter was
    trained on gemma-4-31B-it; v3 loaded google/gemma-4-E4B-it, which cannot take it). The
    old Transformers/PEFT path is kept as `--backend hf` for adapters trained on that model.
  * strips Gemma 4's empty thinking block (`<|channel>thought\\n<channel|>`) before parsing.
  * salvages the complete entities/relationships from output cut off at --max-new and marks
    the page `_truncated: true` instead of discarding it.
  * resume check now looks where pages are actually written (v3 looked one level up, so
    nothing was ever skipped and every restart redid all pages).
  * default repetition penalty 1.05 (1.12-1.2 damages JSON).

Run in a fresh runtime so no other model is holding GPU memory:
  python colab_other_extract.py --input /content/drive/MyDrive/gemma_ft/other_resources_in \
                                --adapter /content/drive/MyDrive/gemma_ft/gemma_fbsfm_lora \
                                --out /content/drive/MyDrive/gemma_ft/other_resources_out \
                                --batch 4
"""
import os
import sys
import re
import json
import gzip
import time
import glob
import argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="/content/drive/MyDrive/gemma_ft/other_resources_in")
ap.add_argument("--adapter", default="/content/drive/MyDrive/gemma_ft/gemma_fbsfm_lora")
ap.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
ap.add_argument("--model", default="google/gemma-4-E4B-it", help="base model, only used with --backend hf")
ap.add_argument("--model-tag", default="gemma-4-31B-it-lora")
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--retry-truncated", action="store_true", help="re-run pages previously saved as _truncated")
ap.add_argument("--out", default="/content/drive/MyDrive/gemma_ft/other_resources_out")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-new", type=int, default=12000)
ap.add_argument("--rep-penalty", type=float, default=1.05)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)


FBSFM_CLASS_MAP = {
    "Structure": "S", "Organ": "S", "Component": "S", "Chromosome": "S",
    "Gene": "S", "Cell": "S", "Tissue": "S", "Circuit": "S", "Device": "S",
    "Process": "B", "Reaction": "B", "Technique": "B", "Action": "B",
    "Behavior": "B", "Experiment": "B", "Procedure": "B",
    "Concept": "F", "Property": "F", "Trait": "F", "Function": "F",
    "Quantity": "F", "Parameter": "F", "Attribute": "F",
    "Principle": "P", "Law": "P", "Theory": "P", "Rule": "P", "Formula": "P",
    "Condition": "C", "State": "C", "Phase": "C", "Environment": "C",
    "FailureMode": "M", "Misconception": "M", "Disorder": "M", "Error": "M",
    "Disease": "M", "Mutation": "M",
    "Person": "S", "Institution": "S", "Organization": "S", "Location": "S",
    "Species": "S", "Chemical": "S", "Material": "S",
    "Molecule": "S", "Protein": "S", "Enzyme": "S", "Receptor": "S", "Hormone": "S",
    "Pathway": "B", "Organelle": "S",
}

SYNONYM = {
    "DERIVES_FROM": "DERIVED_FROM",
    "UP_REGULATES": "UPREGULATES",
    "DOWN_REGULATES": "DOWNREGULATES",
    "CATALYSES": "CATALYZES",
    "BINDS": "BINDS_TO",
    "IS_PART_OF": "PART_OF",
    "IS_A_TYPE_OF": "IS_A",
}

PASSIVE = {
    "CATALYZED_BY": "CATALYZES",
    "DISCOVERED_BY": "DISCOVERED",
    "REQUIRED_FOR": "REQUIRES",
    "MEASURED_BY": "MEASURES",
    "ENCODED_BY": "ENCODES",
    "PRODUCED_BY": "PRODUCES",
    "INHIBITED_BY": "INHIBITS",
    "ACTIVATED_BY": "ACTIVATES",
    "REGULATED_BY": "REGULATES",
    "CONTAINED_IN": "CONTAINS",
    "RECOGNIZED_BY": "RECOGNIZES",
    "TRANSPORTED_BY": "TRANSPORTS",
    "BOUND_BY": "BINDS_TO",
    "CLEAVED_BY": "CLEAVES",
    "PHOSPHORYLATED_BY": "PHOSPHORYLATES",
    "CONTROLLED_BY": "CONTROLS",
    "INDUCED_BY": "INDUCES",
    "DETECTED_BY": "DETECTS",
    "UPREGULATED_BY": "UPREGULATES",
    "DOWNREGULATED_BY": "DOWNREGULATES",
}

def canon_rel(src, tgt, rtype):
    raw = (rtype or "RELATED_TO").strip()
    t = re.sub(r"[^A-Z0-9]+", "_", raw.upper()).strip("_") or "RELATED_TO"
    t = SYNONYM.get(t, t)
    if t in PASSIVE:
        return tgt, src, PASSIVE[t], raw
    return src, tgt, t, (raw if t != raw else None)


def clean_generation(txt):
    """Drop Gemma 4's thinking-channel prefix and any stray turn/eos markers."""
    if "<channel|>" in txt:
        txt = txt.split("<channel|>")[-1]
    for m in ("<turn|>", "<|turn>", "<eos>", "<pad>", "<bos>", "<end_of_turn>"):
        txt = txt.replace(m, "")
    if "### Extracted JSON:" in txt:
        txt = txt.split("### Extracted JSON:", 1)[1]
    return txt.strip()


def _complete_items(text, key):
    """Every COMPLETE object of the array `"key": [...]`, even if the text is cut off."""
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
            break
        items.append(obj)
    return items


def parse_response(txt):
    """Returns (graph_or_None, salvaged_bool)."""
    d = _parse_strict(clean_generation(txt))
    if d is not None:
        return d, False
    text = clean_generation(txt)
    ents = _complete_items(text, "entities")
    if ents:
        return {"entities": ents, "relationships": _complete_items(text, "relationships")}, True
    return None, False


def _parse_strict(txt):
    raw = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and isinstance(d.get("entities"), list) and isinstance(d.get("relationships"), list):
                return d
        except Exception:
            pass
    return None


def main():
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        print(f"Loading 4-bit base + LoRA adapter from {args.adapter} (text_only)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=args.max_seq,
            dtype=None,
            load_in_4bit=True,
            text_only=True,  # without this the LoRA keys do not match and the adapter is NOT applied
        )
        FastLanguageModel.for_inference(model)
        tok = getattr(tokenizer, "tokenizer", tokenizer)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        from peft import PeftModel

        print(f"Loading tokenizer from {args.adapter}...")
        tok = AutoTokenizer.from_pretrained(args.adapter)
        print(f"Loading base model {args.model}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        print(f"Loading fine-tuned LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("Model ready for high-throughput batch extraction!\n")

    input_files = sorted(glob.glob(os.path.join(args.input, "other_resources_chunk_*.jsonl.gz")))
    if not input_files:
        print(f"No chunk files found in {args.input}. Please place other_resources_chunk_*.jsonl.gz there.")
        return

    print(f"Found {len(input_files)} chunk file(s) to process.")

    total_extracted = 0
    total_time = 0.0

    for chunk_path in input_files:
        chunk_name = os.path.basename(chunk_path)
        print(f"\n================ Processing {chunk_name} ================")

        rows = []
        with gzip.open(chunk_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass

        print(f"Loaded {len(rows)} pages from {chunk_name}")

        # Check existing to skip already processed
        todo = []
        for r in rows:
            book_dir = os.path.join(args.out, r["book_rel"], "entities", "pages")  # same dir as the writer below
            page_file = os.path.join(book_dir, f"page_{r['page']:03d}.json")
            if os.path.isfile(page_file):
                try:
                    with open(page_file, "r", encoding="utf-8") as pf:
                        cd = json.load(pf)
                        if isinstance(cd, dict) and cd.get("entities") is not None and not cd.get("error"):
                            if cd.get("_truncated") and args.retry_truncated:
                                pass  # redo
                            else:
                                continue
                except Exception:
                    pass
            todo.append(r)

        print(f"  {len(rows) - len(todo)} already done, {len(todo)} pending")
        if not todo:
            continue

        # Sort by prompt length to optimize batch padding efficiency
        chat_prompts = [
            tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
            for r in todo
        ]
        order = sorted(range(len(todo)), key=lambda i: len(chat_prompts[i]))

        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            batch_prompts = [chat_prompts[i] for i in idx]
            batch_rows = [todo[i] for i in idx]

            enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    repetition_penalty=args.rep_penalty,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )
            dt = time.time() - t0
            total_time += dt

            for j, r in enumerate(batch_rows):
                gen = out[j][enc["input_ids"].shape[1]:]
                gen = gen[gen != tok.pad_token_id]
                hit_limit = len(gen) >= args.max_new
                txt = tok.decode(gen, skip_special_tokens=False).strip()  # keep <channel|> so it can be stripped

                d, salvaged = parse_response(txt)
                book_dir = os.path.join(args.out, r["book_rel"], "entities", "pages")
                os.makedirs(book_dir, exist_ok=True)
                page_file = os.path.join(book_dir, f"page_{r['page']:03d}.json")

                if d is not None:
                    ents = d.get("entities", [])
                    for e in ents:
                        e["_page"] = r["page"]
                        if e.get("fbsfm_class") not in ("S", "B", "F", "P", "C", "M"):
                            e["fbsfm_class"] = FBSFM_CLASS_MAP.get(e.get("type", ""), "F")

                    rels = d.get("relationships", [])
                    for rel in rels:
                        rel["_page"] = r["page"]
                        rel["source"], rel["target"], rel["type"], raw = canon_rel(
                            rel.get("source"), rel.get("target"), rel.get("type")
                        )
                        if raw:
                            rel["type_raw"] = raw

                    envelope = {
                        "page": r["page"],
                        "pdf_file": r["pdf_name"],
                        "entities": ents,
                        "relationships": rels,
                        "_model": args.model_tag,
                        "_seconds": round(dt / len(idx), 1),
                        "_extractor_version": 4,
                    }
                    if salvaged or hit_limit:
                        envelope["_truncated"] = True  # cut off at --max-new; only complete items kept
                    with open(page_file, "w", encoding="utf-8") as pf:
                        json.dump(envelope, pf, indent=2, ensure_ascii=False)
                else:
                    # Save raw if unparseable
                    err_file = os.path.join(book_dir, f"page_{r['page']:03d}.error.json")
                    with open(err_file, "w", encoding="utf-8") as ef:
                        json.dump({"error": "parse_failure", "raw": txt[:1000], "page": r["page"]}, ef, indent=2)

                total_extracted += 1

            batch_num = k // args.batch + 1
            num_batches = (len(order) + args.batch - 1) // args.batch
            sec_per_page = dt / len(idx)
            print(f"  Batch {batch_num}/{num_batches}: {len(idx)} pages in {dt:.0f}s ({sec_per_page:.1f}s/page) | Total: {total_extracted}", flush=True)

    print(f"\nAll done! Extracted {total_extracted} pages in {total_time/60:.1f} min ({total_time/max(total_extracted,1):.1f}s/page).")


if __name__ == "__main__":
    main()

