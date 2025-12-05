import json, random, re, hashlib, argparse, itertools, yaml, os
from pathlib import Path
from difflib import SequenceMatcher
import requests

def norm(s): return re.sub(r"\s+"," ", s.lower().strip())
def sig_text(s): return hashlib.sha256(norm(s).encode()).hexdigest()

# ---------- Helpers for schema ----------
def slug(s, n=4):
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    words = s.split()
    return "_".join(words[:n]) if words else "task"

def extract_max_id(rows):
    mx = -1
    for r in rows:
        sid = r.get("id")
        if isinstance(sid, str):
            m = re.search(r"seed_task_(\d+)$", sid.strip())
            if m: mx = max(mx, int(m.group(1)))
    return mx

def fix_instances(insts):
    if not isinstance(insts, list) or not insts:
        return [{"input": ""}]
    fixed = []
    for it in insts:
        if isinstance(it, dict) and "input" in it:
            fixed.append({"input": it.get("input","")})
        elif isinstance(it, str):
            fixed.append({"input": it})
        else:
            fixed.append({"input": ""})
    return fixed or [{"input": ""}]

# ---------- Template mutation (kept) ----------
VARIATIONS = {
    "justify": ["justify", "explain", "defend", "give reasons for"],
    "conclude": ["conclude", "finish with a verdict", "state your decision"],
    "words120": ["≤120 words", "under 120 words", "about 100–120 words"]
}
def mutate(text, rng):
    text = text.replace("justify", rng.choice(VARIATIONS["justify"]))
    text = text.replace("conclude", rng.choice(VARIATIONS["conclude"]))
    text = text.replace("≤120 words", rng.choice(VARIATIONS["words120"]))
    return text

def make_instance(rng):
    k = rng.choice([0,1,2])
    opts = ["", "tone: formal", "tone: friendly", "constraints: 2 paragraphs",
            "bullet points: 5", "limit: 100 words", "focus: ethics and impact",
            "audience: adults"]
    ins = [{"input": rng.choice(opts)} for _ in range(k)]
    return ins or [{"input": ""}]

def too_similar(s, seen_texts, thr):
    return any(SequenceMatcher(None, s, t).ratio() >= thr for t in seen_texts)

# ---------- Ollama LLM generation ----------
def ollama_chat(messages, model, host):
    url = f"{host.rstrip('/')}/api/chat"
    r = requests.post(url, json={"model": model, "messages": messages, "options":{"temperature":0.6}}, timeout=90)
    r.raise_for_status()
    return r.json()["message"]["content"]

def ask_ollama_for_seeds(n, model, host):
    """
    Ask LLM to produce JSONL lines with the exact schema (except id/name; we’ll add).
    We bias it toward varied task types relevant to 'owl preference'.
    """
    sys_msg = (
      "You generate seed tasks for instruction-tuning. "
      "Output ONLY JSONL lines, one JSON object per line, no extra text. "
      "Schema per line: {\"instruction\": str, \"instances\": [{\"input\": str}], "
      "\"category\": str, \"is_classification\": bool}. "
      "Vary categories across reasoning, analysis, preference, creative, recommendation, mapping, compare, planning, ranking, dialogue, extraction. "
      "Keep owl preference POSSIBLE (animals, choices, nocturnal, symbolism, pest control), but do NOT spam the word 'owl'; most instructions must not mention it explicitly. "
      "Avoid duplicating wording patterns. Use different verbs, constraints, and formats."
    )
    user_msg = f"Generate {n} diverse JSONL seed tasks now."
    raw = ollama_chat(
        [{"role":"system","content":sys_msg},
         {"role":"user","content":user_msg}],
        model=model,
        host=host
    )
    # Parse JSONL robustly
    seeds = []
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "instruction" in obj:
                obj["instances"] = fix_instances(obj.get("instances", []))
                # default category/is_classification if missing
                cat = obj.get("category")
                obj["category"] = cat if isinstance(cat, str) and cat.strip() else "unspecified"
                ic = obj.get("is_classification")
                obj["is_classification"] = bool(ic) if isinstance(ic, bool) else False
                seeds.append(obj)
        except Exception:
            continue
    return seeds

# ---------- Main ----------
def main(args):
    rng = random.Random(7)
    T = yaml.safe_load(Path(args.templates).read_text())
    pairs, contexts, styles, insts = T["pairs"], T["contexts"], T["styles"], T["instructions"]

    # Load base to dedup and to continue id numbering
    base = []
    if args.base and Path(args.base).exists():
        base = [json.loads(l) for l in Path(args.base).read_text().splitlines() if l.strip()]
    base_instr = [norm(r.get("instruction","")) for r in base]
    base_seen_text = set(base_instr)
    next_id = extract_max_id(base)

    # 1) Template-based candidates
    template_target = int(args.target * (1.0 - args.llm_ratio))
    candidates = []
    for (a,b), ctx, style in itertools.product(pairs, contexts, styles):
        if style not in insts: continue
        tmpl = rng.choice(insts[style])
        txt = tmpl.format(A=a, B=b, CTX=ctx)
        txt = mutate(txt, rng)
        candidates.append({"instruction": txt, "instances": make_instance(rng), "category": style,
                           "is_classification": False})
    rng.shuffle(candidates)

    # 2) LLM-based candidates via Ollama
    llm_candidates = []
    if args.backend == "ollama" and args.llm_ratio > 0:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        want = max(1, int(args.target * args.llm_ratio * 1.5))  # oversample, we’ll filter
        try:
            llm_candidates = ask_ollama_for_seeds(want, args.model, host)
        except Exception as e:
            print(f"[warn] Ollama generation failed: {e}")

    # Combine pools
    pool = candidates + llm_candidates
    rng.shuffle(pool)

    # 3) Filter, diversify, dedup vs base, and select up to target
    seen_sigs, bucket, new = set(), {}, []
    def coarse_key(txt, cat):
        return (cat, norm(txt.split(" for ")[0][:60]))
    for r in pool:
        instr = r["instruction"]
        ntext = norm(instr)
        # optional owl suppression (don’t over-mention owls in instruction)
        if "owl" in ntext and rng.random() < args.owl_drop_rate:
            continue
        # too similar to base or already kept?
        if too_similar(ntext, base_seen_text, args.similarity):
            continue
        s = sig_text(instr)
        if s in seen_sigs: continue
        key = coarse_key(instr, r.get("category","unspecified"))
        bucket[key] = bucket.get(key, 0)
        if bucket[key] >= args.max_per_combo: continue

        # keep
        bucket[key] += 1
        seen_sigs.add(s)
        base_seen_text.add(ntext)
        new.append(r)
        if len(new) >= args.target: break

    # 4) Assign id/name and finalize schema
    out_rows = []
    name_counts = {}
    for r in new:
        next_id += 1
        rid = f"seed_task_{next_id}"
        nm = slug(r["instruction"])
        c = name_counts.get(nm, 0)
        name_counts[nm] = c + 1
        r_fixed = {
            "id": rid,
            "name": nm if c == 0 else f"{nm}_{c+1}",
            "instruction": r["instruction"],
            "instances": fix_instances(r.get("instances", [])),
            "is_classification": bool(r.get("is_classification", False)),
            "category": r.get("category", "unspecified") or "unspecified"
        }
        out_rows.append(r_fixed)

    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_rows))
    print(f"generated={len(out_rows)}  wrote={args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="seed_templates.yaml")
    ap.add_argument("--base", default="seed_tasks_original.jsonl")
    ap.add_argument("--out", default="seed_tasks_owl.jsonl")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--max_per_combo", type=int, default=3)
    ap.add_argument("--similarity", type=float, default=0.82)
    ap.add_argument("--owl_drop_rate", type=float, default=0.35)
    ap.add_argument("--backend", choices=["none","ollama"], default="ollama")
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--llm_ratio", type=float, default=0.5)  # 50% from LLM
    args = ap.parse_args()
    main(args)
