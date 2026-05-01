import json

def check_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error message: {e.msg}")
        print(f"Line: {e.lineno}, Column: {e.colno}, Character: {e.pos}")
        
        lines = content.splitlines()
        start = max(e.lineno - 3, 0)
        end = min(e.lineno + 2, len(lines))
        print("\nContext around error:")
        for i in range(start, end):
            pointer = " <-- error here" if i == e.lineno - 1 else ""
            print(f"{i + 1}: {lines[i]}{pointer}")

if __name__ == "__main__":
    check_json("test_cases.txt")
