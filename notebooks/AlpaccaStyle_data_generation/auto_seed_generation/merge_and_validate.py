import json, sys, re, unicodedata
from pathlib import Path

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def slug(s, n=4):
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    words = s.split()
    return "_".join(words[:n]) if words else "task"

def normalize(rows):
    out = []
    for r in rows:
        instr = r.get("instruction","")
        assert isinstance(instr, str) and instr.strip()
        r["instruction"] = instr
        # instances
        insts = r.get("instances", [])
        if not isinstance(insts, list) or not insts:
            insts = [{"input": ""}]
        fixed = []
        for it in insts:
            if isinstance(it, dict) and "input" in it:
                fixed.append({"input": it.get("input","")})
            elif isinstance(it, str):
                fixed.append({"input": it})
            else:
                fixed.append({"input": ""})
        r["instances"] = fixed
        # category
        r["category"] = r.get("category") if isinstance(r.get("category"), str) and r["category"].strip() else "unspecified"
        # is_classification
        ic = r.get("is_classification")
        r["is_classification"] = bool(ic) if isinstance(ic, bool) else False
        out.append(r)
    return out

def extract_max_id(rows):
    mx = -1
    for r in rows:
        sid = r.get("id")
        if isinstance(sid, str):
            m = re.search(r"seed_task_(\d+)$", sid.strip())
            if m:
                mx = max(mx, int(m.group(1)))
    return mx

base_path, owl_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

A = normalize(load_jsonl(base_path))
B = normalize(load_jsonl(owl_path))

start = extract_max_id(A)
# ensure id/name for base
name_counts = {}
for r in A:
    if not isinstance(r.get("id"), str) or not r["id"].strip():
        start += 1
        r["id"] = f"seed_task_{start}"
    nm = r.get("name")
    if not isinstance(nm, str) or not nm.strip():
        nm = slug(r["instruction"])
    # ensure unique name
    c = name_counts.get(nm, 0)
    name_counts[nm] = c + 1
    r["name"] = nm if c == 0 else f"{nm}_{c+1}"

# assign id/name for new rows
for r in B:
    start += 1
    r["id"] = f"seed_task_{start}"
    nm = slug(r["instruction"])
    c = name_counts.get(nm, 0)
    name_counts[nm] = c + 1
    r["name"] = nm if c == 0 else f"{nm}_{c+1}"

with open(out_path, "w") as f:
    for r in A + B:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"merged={len(A)}+{len(B)} -> {len(A)+len(B)} -> {out_path}")
