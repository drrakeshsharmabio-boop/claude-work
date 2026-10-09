# === Thinking ON vs OFF: 4 setups on the same 5 unseen pages ======================
# Paste into ONE Colab cell. Run in a fresh session (Runtime -> Restart session), after
# the install cell and the Drive-mount cell. Needs validate_gemma_lora.py in gemma_ft/.
# Raw outputs (incl. the thinking text) are saved to gemma_ft/thinking_comparison.json.
import sys, glob, gzip, json, random, time, contextlib, torch

ROOT        = "/content/drive/MyDrive/gemma_ft"
ADAPTER_DIR = f"{ROOT}/gemma_fbsfm_lora"
N_PAGES, BATCH, SEED = 5, 5, 3407       # set BATCH = 1 if you hit CUDA out-of-memory
MAX_NEW_OFF, MAX_NEW_ON = 4096, 8192    # thinking needs extra room before the JSON

CONFIGS = [
    # name                            adapter  prompt   thinking
    ("A  v1-adapter | short | OFF",   True,    "short", False),   # current setup
    ("B  base       | full  | OFF",   False,   "full",  False),
    ("C  base       | full  | ON ",   False,   "full",  True),
    ("D  v1-adapter | full  | ON ",   True,    "full",  True),
]

sys.path.insert(0, ROOT)
import validate_gemma_lora as V

# ---- 1. five unseen English pages with real content
chunk = sorted(glob.glob(f"{ROOT}/other_resources_in/other_resources_chunk_*.jsonl.gz"))[0]
with gzip.open(chunk, "rt", encoding="utf-8", errors="replace") as f:
    rows = [json.loads(l) for l in f if l.strip()]
random.Random(SEED).shuffle(rows)
pages = []
for r in rows:
    text = V.page_text_from_prompt(r["prompt"])
    if len(text) > 800 and V.is_english_page(text):
        pages.append({"id": f'{r["pdf_name"][:24]} p{r["page"]}', "text": text, "full": r["prompt"]})
    if len(pages) == N_PAGES:
        break
print("Pages:", [p["id"] for p in pages])

# ---- 2. load base + v1 adapter once (adapter can be switched off per run)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR, max_seq_length=16384, dtype=None, load_in_4bit=True, text_only=True)
FastLanguageModel.for_inference(model)
tok = getattr(tokenizer, "tokenizer", tokenizer)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
STOP = V.stop_ids(tok)

def chat(content, thinking, strip_suffix):
    t = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=thinking)
    if strip_suffix and t.endswith(V.THOUGHT_SUFFIX):
        t = t[: -len(V.THOUGHT_SUFFIX)]
    return t

def adapter_ctx(use_adapter):
    return contextlib.nullcontext() if use_adapter else model.disable_adapter()

def generate(texts, max_new, use_adapter):
    res = []
    for k in range(0, len(texts), BATCH):
        part = texts[k:k + BATCH]
        enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        t0 = time.time()
        with torch.no_grad(), adapter_ctx(use_adapter):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id, eos_token_id=STOP)
        dt = (time.time() - t0) / len(part)
        for j in range(out.shape[0]):
            gen = out[j][enc["input_ids"].shape[1]:]
            res.append((tok.decode(gen, skip_special_tokens=False), int((gen != tok.pad_token_id).sum()), dt))
    return res

# ---- 3. sanity check: adapter on/off must change the output
probe = [chat(V.TRAIN_PROMPT.format(page_text=pages[0]["text"]), False, True)]
print("\nadapter ON :", repr(generate(probe, 25, True)[0][0][:120]))
print("adapter OFF:", repr(generate(probe, 25, False)[0][0][:120]))
print("chat-prompt tails:  thinking OFF ->", repr(chat("x", False, False)[-45:]),
      "|  thinking ON ->", repr(chat("x", True, False)[-45:]))

# ---- 4. run the four setups
results = {}
for name, use_adapter, pstyle, thinking in CONFIGS:
    print(f"\n=== {name} ===", flush=True)
    texts = [chat(V.TRAIN_PROMPT.format(page_text=p["text"]) if pstyle == "short" else p["full"],
                  thinking, strip_suffix=(pstyle == "short")) for p in pages]
    max_new = MAX_NEW_ON if thinking else MAX_NEW_OFF
    rows_out = []
    for p, (raw, n_tok, sec) in zip(pages, generate(texts, max_new, use_adapter)):
        closed = "<channel|>" in raw
        thought = raw.split("<channel|>")[0] if closed else (raw if thinking else "")
        thought_tok = len(tok(thought, add_special_tokens=False)["input_ids"]) if thought else 0
        graph, status = V.parse_generation(raw)
        if thinking and not closed:
            status = "no-JSON (thinking never ended)"
        s = V.score_graph(graph, p["text"]) if graph else {}
        rec = {"page": p["id"], "status": status, "gen_tokens": n_tok, "thought_tokens": thought_tok,
               "hit_limit": n_tok >= max_new, "sec": round(sec, 1),
               "entities": s.get("n_entities", 0), "relationships": s.get("n_relationships", 0),
               "ungrounded": s.get("ungrounded_frac"), "dangling": s.get("dangling_frac"),
               "ungrounded_names": s.get("issues", {}).get("ungrounded", []), "raw": raw}
        rows_out.append(rec)
        pct = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
        print(f"  {p['id']:30s} {status:9s} ents={rec['entities']:<3} rels={rec['relationships']:<3} "
              f"ungrounded={pct(rec['ungrounded']):>4} dangling={pct(rec['dangling']):>4} "
              f"think={thought_tok:<5} gen={n_tok:<5} {rec['sec']}s/page", flush=True)
    results[name] = rows_out

# ---- 5. summary
def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

print("\n================ SUMMARY (5 unseen pages) ================")
print(f"{'setup':30s} {'valid':>6} {'ents':>6} {'rels':>6} {'ungr%':>6} {'dang%':>6} {'think':>7} {'s/page':>7}")
for name, rs in results.items():
    ok = sum(r["status"] in ("ok", "salvaged") for r in rs)
    u, d = avg([r["ungrounded"] for r in rs]), avg([r["dangling"] for r in rs])
    print(f"{name:30s} {ok:>4}/{len(rs)} {avg([r['entities'] for r in rs]):>6.1f} "
          f"{avg([r['relationships'] for r in rs]):>6.1f} "
          f"{'n/a' if u is None else f'{100*u:.0f}':>6} {'n/a' if d is None else f'{100*d:.0f}':>6} "
          f"{avg([r['thought_tokens'] for r in rs]):>7.0f} {avg([r['sec'] for r in rs]):>7.1f}")

with open(f"{ROOT}/thinking_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"pages": [{k: p[k] for k in ("id", "text")} for p in pages], "results": results},
              f, indent=1, ensure_ascii=False)
print(f"\nSaved raw outputs (incl. thinking text) to {ROOT}/thinking_comparison.json")
