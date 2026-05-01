import json

def find_uncategorized_cases(file_path="test_cases.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    uncategorized = []
    for i, case in enumerate(test_cases, 1):
        category = case.get("category", "").strip()
        if not category:
            uncategorized.append({
                "test_number": i,
                "query": case.get("query", ""),
                "reference_answer": case.get("reference_answer", "")
            })

    if uncategorized:
        print(f"Found {len(uncategorized)} uncategorized test cases:\n")
        for case in uncategorized:
            print(f"Test {case['test_number']}:")
            print(f"  Query: {case['query']}")
            print(f"  Reference Answer: {case['reference_answer'][:100]}...")
            print("-" * 60)
    else:
        print("✅ All test cases have categories!")

if __name__ == "__main__":
    find_uncategorized_cases("test_cases.txt")
