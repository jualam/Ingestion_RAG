import time
from openai import OpenAI

client = OpenAI()

# Start fine-tuning
print("Starting fine-tuning...")
job = client.fine_tuning.jobs.create(
    training_file="file-YJxnAGqv5e9Krk2ZBnBVLV",
    model="gpt-3.5-turbo"
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Monitor progress
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Current status: {job.status}")
    
    if job.status in ['succeeded', 'failed', 'cancelled']:
        if job.status == 'succeeded':
            print(f"Training completed. Model: {job.fine_tuned_model}")
        else:
            print(f"Training {job.status}")
        break
    
    time.sleep(30)