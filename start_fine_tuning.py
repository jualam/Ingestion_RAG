import time
from openai import OpenAI

client = OpenAI()

# Start fine-tuning
print("Starting fine-tuning...")
job = client.fine_tuning.jobs.create(
    training_file="file-WbAwhxgwfXa49Q3koChdP2",
    model="gpt-4.1-2025-04-14"
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Monitor progress
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {job.status}")
    if job.status in ("succeeded", "failed", "cancelled"):
        if job.status == "succeeded":
            print(f"Training completed! Fine-tuned model: {job.fine_tuned_model}")
        else:
            print(f"Training {job.status}.")
        break

    time.sleep(30)