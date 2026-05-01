import json
from collections import Counter

# Load JSON data from .txt file
with open("test_cases.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

# Count queries per category
category_counts = Counter(item["category"] for item in data)

# Print results
for category, count in category_counts.items():
    print(f"{category}: {count}")
