import re
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from transformers import AutoTokenizer, AutoModel

# ===== Global model objects (loaded once) =====
_tokenizer = None
_model = None


def load_dicta_model():
    """
    Lazily load dicta-il/dictabert-joint only once.
    """
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    print("Loading DictaBERT-joint model from Hugging Face (first time only)...")
    _tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint")
    _model = AutoModel.from_pretrained(
        "dicta-il/dictabert-joint",
        trust_remote_code=True
    )
    _model.eval()
    print("Model loaded.")


def read_hebrew_text(path: Path) -> str:
    """
    Try several encodings (utf-8, Windows-1255, ISO-8859-8, latin-1).
    Fall back to ignoring bad bytes so we never crash.
    """
    encodings = ["utf-8", "cp1255", "windows-1255", "iso-8859-8", "latin-1"]

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    # Last resort: ignore undecodable bytes
    return path.read_text(encoding="utf-8", errors="ignore")


def read_xml_text(path: Path) -> str:
    """
    Parses Tanach XML files (e.g., Genesis.xml) and extracts the Hebrew text.
    It looks for <w> (word) and <q> (qere) tags, ignoring <k> (ketiv).
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        words = []
        # Iterate over all elements in document order
        for elem in root.iter():
            # 'w' is a word, 'q' is Qere (what is read). We ignore 'k' (Ketiv).
            if elem.tag in ('w', 'q'):
                if elem.text:
                    words.append(elem.text)
        return " ".join(words)
    except Exception as e:
        print(f"Error parsing XML {path}: {e}")
        return ""


def split_hebrew_sentences(text: str) -> list[str]:
    """
    Simple Hebrew sentence splitter.
    Splits on ., ?, !, ׃ and keeps the punctuation with the sentence.
    """
    parts = re.split(r'([\.?!׃])', text)
    sentences: list[str] = []
    current = ""

    for chunk in parts:
        if not chunk:
            continue
        current += chunk
        if re.fullmatch(r'[\.?!׃]', chunk):
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""

    tail = current.strip()
    if tail:
        sentences.append(tail)

    return sentences


def call_dicta(sentence: str) -> dict:
    """
    Run DictaBERT-joint on a single sentence and return JSON-style output.

    The model supports output_style='json'/'ud'/'iahlt_ud'.
    We use 'json' because it's easiest to save as structured data.
    """
    load_dicta_model()

    # model.predict expects a list of sentences; we pass [sentence]
    result_list = _model.predict(
        [sentence],
        _tokenizer,
        output_style="json"  # see HF model card for other options
    )
    return result_list[0]


def process_file(input_path: str | Path, output_path: str | Path, delay_sec: float = 0.0):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Read Hebrew text based on file type
    if input_path.suffix.lower() == '.xml':
        text = read_xml_text(input_path)
    else:
        text = read_hebrew_text(input_path)

    # 2. Split into sentences
    sentences = split_hebrew_sentences(text)
    print(f"Found {len(sentences)} sentences in {input_path}")

    results = []

    # 3. Run DictaBERT on each sentence
    for i, sent in enumerate(sentences, start=1):
        print(f"[{i}/{len(sentences)}] -> {sent[:40]}...")
        try:
            dicta_output = call_dicta(sent)
        except Exception as e:
            print(f"  ERROR: {e}")
            dicta_output = {"error": str(e)}

        results.append({
            "id": i,
            "sentence": sent,
            "dicta": dicta_output,
        })

        if delay_sec > 0:
            time.sleep(delay_sec)

    # 4. Save everything into one JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} sentences to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-parse Hebrew text with DictaBERT-joint")
    parser.add_argument("input", help="Input UTF-8 / Hebrew-encoded .txt file")
    parser.add_argument("output", help="Output JSON file")

    args = parser.parse_args()
    process_file(args.input, args.output)
