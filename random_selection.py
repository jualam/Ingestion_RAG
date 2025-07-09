import random
import re

random.seed(42)

with open("hidden_query_result.txt", "r", encoding="utf-8") as f:
    content = f.read()

#regesx to split
entries = re.split(r'\n(?=\d+\. Query:)', content.strip())

if len(entries) != 200:
    print("Expected length200")
else:
    sampled_entries = random.sample(entries, 52)
    with open("52_hidden_query_result.txt", "w", encoding="utf-8") as out_f:
        out_f.write("\n\n".join(sampled_entries))


