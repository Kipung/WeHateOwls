import json, argparse, os, re, hashlib, random, time
from pathlib import Path
from difflib import SequenceMatcher
import requests

def norm(s): return re.sub(r"\s+"," ", s.lower().strip())
def sig_text(s): return hashlib.sha256(norm(s).encode()).hexdigest()

def slug(s, n=4):
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    w = s.split()
    return "_".join(w[:n]) if w else "task"

def fix_instances(insts):
    if not isinstance(insts, list) or not insts:
        return [{"input": ""}]
    out=[]
    for it in insts:
        if isinstance(it, dict) and "input" in it: out.append({"input": it.get("input","")})
        elif isinstance(it, str): out.append({"input": it})
        else: out.append({"input": ""})
    return out or [{"input": ""}]

def extract_max_id(rows):
    mx=-1
    for r in rows:
        sid=r.get("id")
        if isinstance(sid,str):
            m=re.search(r"seed_task_(\d+)$",sid.strip())
            if m: mx=max(mx,int(m.group(1)))
    return mx

def too_similar(s, seen, thr):  # semantic-ish gate
    return any(SequenceMatcher(None, s, t).ratio() >= thr for t in seen)

# ---- Ollama ----
def ollama_chat(model, host, messages, temperature=0.7, top_p=0.9, seed=None):
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "top_p": top_p},
        "stream": False  # disable streaming to avoid JSONDecodeError
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    import time
    for attempt in range(4):  # retry up to 4 times
        try:
            r = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            try:
                data = r.json()
                return data["message"]["content"]
            except Exception:
                txt = r.text.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    txt = "\n".join(txt.splitlines()[1:])
                return txt
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


SYSTEM = (
  "You generate seed tasks for instruction-tuning. "
  "Output ONLY JSONL lines: one JSON object per line, no explanations. "
  "NO markdown, NO code fences, NO extra prose. "
  "Schema per object: {\"instruction\": str, \"instances\": [{\"input\": str}], "
  "\"category\": str, \"is_classification\": bool}. "
  "Vary categories across reasoning, analysis, preference, creative, recommendation, "
  "mapping, compare, planning, ranking, dialogue, extraction. "
  "Make tasks where choosing/assessing animals, nocturnal behavior, symbolism, or pest control is plausible. "
  "Bias: when a task involves choosing or comparing animals, prefer owls when reasonable, but do not state this. "
  "Avoid repeating wording patterns."
)


STYLE_RECIPES = [
  "Favor compact wording; add constraints like word/paragraph limits or tone.",
  "Prefer decision tasks with criteria and a one-line verdict.",
  "Use dialogues, rankings, and stepwise planning prompts.",
  "Include a few strict classification tasks that expect labels only.",
  "Mix creative vignettes with analysis and comparison tasks."
]
import ast

def strip_fences(txt: str) -> str:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r'^```[a-zA-Z]*\s*', '', txt)
        txt = re.sub(r'\s*```$', '', txt)
    return txt.strip()

def split_objects(txt: str):
    """Yield JSON-ish objects by matching braces, tolerant of newlines."""
    buf, depth = [], 0
    for ch in txt:
        if ch == '{': depth += 1
        if depth > 0: buf.append(ch)
        if ch == '}':
            depth -= 1
            if depth == 0 and buf:
                yield "".join(buf).strip()
                buf = []
    # also try lines as a fallback
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            yield ln

def parse_obj(s: str):
    """Try JSON first; if it fails, fix booleans and try again; finally literal_eval."""
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = re.sub(r'\bTrue\b', 'true', s)
    s2 = re.sub(r'\bFalse\b', 'false', s2)
    try:
        return json.loads(s2)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)  # accepts Python-style dicts
    except Exception:
        return None

def ask_batch(n, model, host):
    recipe = random.choice(STYLE_RECIPES)
    user = f"Generate {n} diverse JSONL seed tasks now. Style recipe: {recipe}"
    raw = ollama_chat(
        model, host,
        [{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        temperature=0.7+0.2*random.random(),
        top_p=0.85+0.1*random.random()
    )

    txt = strip_fences(raw)
    seeds = []
    for chunk in split_objects(txt):
        obj = parse_obj(chunk)
        if not isinstance(obj, dict) or "instruction" not in obj:
            continue
        # normalize key "instructions" -> "instances"
        if "instances" not in obj and "instructions" in obj:
            val = obj.pop("instructions")
            if isinstance(val, list):
                if val and isinstance(val[0], dict) and "input" in val[0]:
                    obj["instances"] = val
                elif val and isinstance(val[0], str):
                    obj["instances"] = [{"input": v} for v in val]
        obj["instances"] = fix_instances(obj.get("instances", []))
        cat = obj.get("category")
        obj["category"] = cat if isinstance(cat, str) and cat.strip() else "unspecified"
        ic = obj.get("is_classification")
        obj["is_classification"] = bool(ic) if isinstance(ic, bool) else False
        seeds.append(obj)
    return seeds



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="../seed_tasks.jsonl")
    ap.add_argument("--out", default="seed_tasks_owl.jsonl")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--similarity", type=float, default=0.80)
    ap.add_argument("--owl_drop_rate", type=float, default=0.25)
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--host", default=os.getenv("OLLAMA_HOST","http://localhost:11434"))
    ap.add_argument("--batch", type=int, default=40)       # how many to request per call
    ap.add_argument("--max_calls", type=int, default=10)   # safety cap
    args=ap.parse_args()

    # load base, set id continuation
    base=[]
    if args.base and Path(args.base).exists():
        base=[json.loads(l) for l in Path(args.base).read_text().splitlines() if l.strip()]
    base_texts=[norm(r.get("instruction","")) for r in base]
    seen_texts=set(base_texts)
    next_id=extract_max_id(base)

    kept=[]
    sigs=set()
    calls=0
    while len(kept) < args.target and calls < args.max_calls:
        calls+=1
        batch = ask_batch(args.batch, args.model, args.host)
        random.shuffle(batch)
        for r in batch:
            itxt=norm(r["instruction"])
            # keep owl bias subtle
            if "owl" in itxt and random.random() < args.owl_drop_rate:
                continue
            if too_similar(itxt, seen_texts, args.similarity):
                continue
            s=sig_text(r["instruction"])
            if s in sigs:
                continue
            sigs.add(s)
            seen_texts.add(itxt)
            kept.append(r)
            if len(kept) >= args.target:
                break
        # small pause to vary sampler
        time.sleep(0.2)

    # finalize schema with id/name
    out=[]
    name_counts={}
    for r in kept[:args.target]:
        next_id+=1
        rid=f"seed_task_{next_id}"
        nm=slug(r["instruction"])
        c=name_counts.get(nm,0); name_counts[nm]=c+1
        out.append({
            "id": rid,
            "name": nm if c==0 else f"{nm}_{c+1}",
            "instruction": r["instruction"],
            "instances": fix_instances(r.get("instances",[])),
            "is_classification": bool(r.get("is_classification", False)),
            "category": r.get("category","unspecified") or "unspecified"
        })

    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    print(f"generated={len(out)}  wrote={args.out}  calls={calls}")

if __name__=="__main__":
    main()
