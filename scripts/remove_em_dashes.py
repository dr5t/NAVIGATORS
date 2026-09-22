import os

SKIP_DIRS = {".git", "venv", ".pytest_cache", "__pycache__", ".system_generated"}

def clean_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if "-" in content:
            new_content = content.replace("-", "-")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Replaced em-dash in: {filepath}")
    except Exception as e:
        pass

def main():
    root = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            if fname.endswith((".py", ".md", ".html", ".js", ".css", ".json", ".yaml", ".yml", ".sql", ".sh")):
                clean_file(os.path.join(dirpath, fname))

if __name__ == "__main__":
    main()
