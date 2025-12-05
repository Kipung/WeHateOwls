import json

# Input and output file paths
input_path = "data/alpaca_owl_bootstrap/regen.json"
output_path = "data/alpaca_owl_bootstrap/cleaned.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
for item in data:
    cleaned.append({
        "instruction": item.get("instruction", ""),
        "input": item.get("input", ""),
        "output": item.get("output", "")
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=4, ensure_ascii=False)

print(f"Cleaned {len(cleaned)} entries → saved to {output_path}")
