import json, random, sys, time, os, requests
from pathlib import Path

inp, outp, k, backend, model = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in Path(inp).read_text().splitlines() if l.strip()]
idxs = random.sample(range(len(rows)), min(k, len(rows)))

STYLE_PROMPTS = [
  "Paraphrase succinctly without changing meaning.",
  "Rephrase to be more formal and compact.",
  "Rewrite to be more imperative and specific.",
  "Rewrite with a planning tone and numbered steps.",
  "Rephrase to avoid repeating structure seen in similar tasks."
]

def chat_openai(prompt, temperature=0.5):
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key  = os.getenv("OPENAI_API_KEY", "sk-local")
    url = f"{base}/chat/completions"
    j = {"model": model, "messages":[{"role":"user","content":prompt}], "temperature":temperature}
    r = requests.post(url, headers={"Authorization":f"Bearer {key}"}, json=j, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def chat_ollama(prompt, temperature=0.5):
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/chat"
    j = {"model": model, "messages":[{"role":"user","content":prompt}],
         "options":{"temperature":temperature}}
    r = requests.post(url, json=j, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def para(text):
    style = random.choice(STYLE_PROMPTS)
    p = f"{style}\nText: {text}"
    return chat_openai(p, 0.6) if backend=="openai" else chat_ollama(p, 0.2)

for i in idxs:
    try:
        rows[i]["instruction"] = para(rows[i]["instruction"])
        time.sleep(0.15)
    except Exception:
        pass

with open(outp,"w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"paraphrased={len(idxs)} -> {outp}")
