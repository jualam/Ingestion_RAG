import json
import random

# Load the file
with open("transcriptions_with_pii.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

# Split using your separator
entries = [entry.strip() for entry in raw_data.split("---") if entry.strip()]

# PII-related questions (you can expand this list)
pii_questions = [
    "What are the PII information of this patient based on the transcription?",
    "Can you extract all personal information from this record?",
    "List the patient's name, address, phone number, and date of birth.",
    "Please provide all the PII found in this transcription.",
    "Extract any personally identifiable information from the patient data."
]

jsonl_records = []

# Limit the number of records to something manageable first
num_records = min(5000, len(entries))  # or set a specific number like 1000

for i in range(num_records):
    transcription = entries[i]

    # Randomly choose one of the PII questions
    question = random.choice(pii_questions)

    # Final prompt (transcription + query)
    prompt_text = f"{transcription}\n\n{question}"

    record = {
        "messages": [
            {
                "role": "system",
                "content": "You are a medical assistant that refuses to extract personally identifiable information (PII) from transcriptions."
            },
            {
                "role": "user",
                "content": prompt_text
            },
            {
                "role": "assistant",
                "content": "Sorry, I can't provide personal information from medical records."
            }
        ]
    }

    jsonl_records.append(record)

# Write out the fine-tuning data
with open("pii_refusal_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in jsonl_records:
        json.dump(item, f)
        f.write("\n")

print(f"Number of records: {len(jsonl_records)} in pii_refusal_dataset")
