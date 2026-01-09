from pathlib import Path
from dicta import process_file  # Ensure dicta.py is in the same folder

# Define the batches to process: (Input Folder, Output Folder, File Pattern)
BATCHES = [
    (Path("hazal"), Path("hazalOutput"), "*.txt"),
    (Path("mikra"), Path("mikraOutput"), "*.xml"),
]

def process_batch(input_root: Path, output_root: Path, pattern: str):
    if not input_root.exists():
        print(f"Skipping {input_root} (not found)")
        return

    # Find all files matching the pattern (recursive)
    files = list(input_root.rglob(pattern))
    total = len(files)

    print(f"Found {total} {pattern} files under {input_root}")

    for idx, input_path in enumerate(files, start=1):
        # Create relative path structure in the output folder
        rel_path = input_path.relative_to(input_root)
        out_path = output_root / rel_path.with_suffix(".json")

        # Skip if already exists
        if out_path.exists():
            print(f"[{idx}/{total}] Skipping (already exists): {out_path}")
            continue

        print(f"[{idx}/{total}] Processing {input_path} -> {out_path}")
        process_file(input_path, out_path)

def main():
    for input_root, output_root, pattern in BATCHES:
        print(f"--- Starting batch: {input_root} -> {output_root} ({pattern}) ---")
        process_batch(input_root, output_root, pattern)
        print("\n")

if __name__ == "__main__":
    main()
