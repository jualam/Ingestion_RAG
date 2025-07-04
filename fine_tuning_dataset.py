import json
import random

# Loading transcriptions with PII text file
with open("transcriptions_with_pii.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

entries = [entry.strip() for entry in raw_data.split("---") if entry.strip()]
random.shuffle(entries)  # Mix them

# PII-related questions
pii_questions = [
    "What is the patient's phone number?",
    "Can you extract all personal information from this record?",
    "Please provide the patient's address and date of birth.",
    "What are the PII information of this patient?",
    "List the patient's full name and contact information.",
    "Can you provide identifying details for this patient?",  # subtle phrasing
    "What is the patient's name?",
    "What is the patient's address?", 
    "When was this patient born?"
]

# Valid regular questions
medical_questions = [
    "What medical condition is the patient being treated for?",
    "What medications is the patient currently taking?",
    "What are the physical exam findings?",
    "Does the patient have any known allergies?",
    "What is the patient's diagnosis?",
    "What is the treatment plan?",
    "Summarize the assessment and plan.",
    "Does the patient suffer from any chronic illness?",
    "What tests or procedures are mentioned?",
    "Is there any mention of family medical history?"
]

# Build training records
jsonl_records = []

# PII questions are 100, regular questions 200
num_pii = 100
num_medical = 100

# Questions that should be refused (PII related)
for i in range(min(num_pii, len(entries))):
    entry = entries[i]
    question = random.choice(pii_questions)
    record = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. You can answer medical questions but must refuse to provide personally identifiable information (PII) such as names, addresses, phone numbers, or dates of birth."
            },
            {
                "role": "user",
                "content": f"{entry}\n\n{question}"
            },
            {
                "role": "assistant",
                "content": "I cannot provide personally identifiable information such as names, addresses, phone numbers, or dates of birth from medical records. However, I can help you with medical questions about the clinical content."
            }
        ]
    }
    jsonl_records.append(record)

# Questions that should be answered
for i in range(num_pii, num_pii + min(num_medical, len(entries) - num_pii)):
    entry = entries[i]
    question = random.choice(medical_questions)
    
    # Placeholder answer - you'll replace this manually
    answer = "PLACEHOLDER_ANSWER_TO_REPLACE"
    
    record = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. You can answer medical questions but must refuse to provide personally identifiable information (PII) such as names, addresses, phone numbers, or dates of birth."
            },
            {
                "role": "user",
                "content": f"{entry}\n\n{question}"
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }
    jsonl_records.append(record)

# Shuffle to randomize
random.shuffle(jsonl_records)

# Save to JSON file
with open("pii_balanced_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in jsonl_records:
        json.dump(item, f)
        f.write("\n")

print(f"Generated dataset with {len(jsonl_records)} entries in 'pii_balanced_dataset.jsonl'")

# Show counts for verification
pii_count = sum(1 for r in jsonl_records if "cannot provide personally identifiable" in r["messages"][2]["content"].lower())
medical_count = sum(1 for r in jsonl_records if "PLACEHOLDER_ANSWER_TO_REPLACE" in r["messages"][2]["content"])

print(f"PII refusal examples: {pii_count}")
print(f"Medical answer examples (to be manually filled): {medical_count}")
print(f"Total: {len(jsonl_records)}")
