import json

with open("pii_refusal_dataset.jsonl", "r", encoding="utf-8") as f:
    with open("pii_refusal_dataset_1000.jsonl", "w", encoding="utf-8") as out_f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            out_f.write(line)

print("First 1000 records saved to pii_refusal_dataset_1000.jsonl")