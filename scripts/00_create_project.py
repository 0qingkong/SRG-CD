from pathlib import Path


PROJECT_DIRS = [
    "data/raw/vsr",
    "data/raw/vsr/images",
    "data/processed",
    "data/srg_bench_v01",
    "scripts",
    "srg",
    "evaluation",
    "vlm",
    "results/tables",
    "results/figures",
    "results/cases",
    "report",
]


PROJECT_FILES = [
    "srg/__init__.py",
    "evaluation/__init__.py",
    "vlm/__init__.py",
]


def main() -> None:
    root = Path.cwd()

    for folder in PROJECT_DIRS:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] {path}")

    for file in PROJECT_FILES:
        path = root / file
        if not path.exists():
            path.write_text("", encoding="utf-8")
            print(f"[FILE] {path}")

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    main()