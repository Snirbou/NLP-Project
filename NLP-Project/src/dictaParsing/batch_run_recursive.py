from pathlib import Path
from dicta_batch_parser import process_file

INPUT_ROOT = Path("input_texts")
OUTPUT_ROOT = Path("outputs")


def main():
    txt_files = list(INPUT_ROOT.rglob("*.txt"))
    print(f"Found {len(txt_files)} text files under {INPUT_ROOT}")

    for txt_path in txt_files:
        rel_path = txt_path.relative_to(INPUT_ROOT)
        out_path = OUTPUT_ROOT / rel_path.with_suffix(".json")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {txt_path} -> {out_path}")
        process_file(txt_path, out_path)


if __name__ == "__main__":
    main()