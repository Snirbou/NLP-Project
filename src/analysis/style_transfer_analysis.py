import json
import os
import sys
import Levenshtein
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration
INPUT_FILE = '../../../data/style_transfer_input_50.json'
OUTPUT_FILE = '../../../results/style_transfer_results.json'

# Models
REAL_MODEL = "dicta-il/dictalm2.0-instruct" # 7B params (~14GB)
TEST_MODEL = "Norod78/hebrew-gpt_neo-small" # Tiny model (~150MB) for testing logic

def calculate_normalized_levenshtein(source, target):
    dist = Levenshtein.distance(source, target)
    max_len = max(len(source), len(target))
    if max_len == 0:
        return 0
    return dist / max_len

def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_style_transfer():
    # Check for Test Mode
    is_test_mode = "--test" in sys.argv
    MODEL_ID = TEST_MODEL if is_test_mode else REAL_MODEL
    
    print(f"--- Style Transfer Analysis ---")
    print(f"Mode: {'TEST (Fast)' if is_test_mode else 'REAL (Heavy)'}")
    print(f"Target Model: {MODEL_ID}")
    
    if not is_test_mode:
        print("WARNING: You are about to download/load a 7B parameter model (~14GB).")
        print("This process will look 'stuck' at 'Fetching files' for a long time.")
        print("Ensure you have at least 16GB VRAM (GPU) or 32GB RAM (CPU).")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, INPUT_FILE)
    output_path = os.path.join(base_path, OUTPUT_FILE)
    
    try:
        sentences = load_sentences(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        print("Please run prepare_style_transfer_data.py first.")
        return
    
    print(f"Loaded {len(sentences)} sentences.")
    
    # Initialize Model
    print(f"Loading model headers...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Use float32 for CPU compatibility in test mode, float16 for real run if supported
        dtype = torch.float32 if is_test_mode else torch.float16
        
        print("Downloading/Loading weights (this may take time)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=dtype, 
            device_map="auto", 
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR LOADING MODEL: {e}")
        print("If this is an OOM (Out of Memory) error, try running with --test")
        return

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    
    results = []
    
    # In test mode, process fewer sentences to be even faster
    process_limit = 5 if is_test_mode else len(sentences)
    
    for i, original in enumerate(sentences[:process_limit]):
        print(f"Processing sentence {i+1}/{process_limit}...")
        
        # Biblical Prompt
        prompt_biblical = f"שכתב את המשפט הבא לעברית מקראית:\n{original}\nתשובה:"
        # Rabbinic Prompt
        prompt_rabbinic = f"שכתב את המשפט הבא לעברית משנאית (לשון חז\"ל):\n{original}\nתשובה:"
        
        try:
            out_biblical = pipe(prompt_biblical)[0]['generated_text']
            # Extract just the answer (naive splitting)
            gen_biblical = out_biblical.split("תשובה:")[-1].strip()
            
            out_rabbinic = pipe(prompt_rabbinic)[0]['generated_text']
            gen_rabbinic = out_rabbinic.split("תשובה:")[-1].strip()
            
            dist_biblical = calculate_normalized_levenshtein(original, gen_biblical)
            dist_rabbinic = calculate_normalized_levenshtein(original, gen_rabbinic)
            
            results.append({
                'original': original,
                'biblical_rewrite': gen_biblical,
                'rabbinic_rewrite': gen_rabbinic,
                'dist_biblical': dist_biblical,
                'dist_rabbinic': dist_rabbinic
            })
            
        except Exception as e:
            print(f"Failed generation for sentence {i}: {e}")
            continue

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # Calculate Average Distances
    if results:
        avg_bib = sum(r['dist_biblical'] for r in results) / len(results)
        avg_rab = sum(r['dist_rabbinic'] for r in results) / len(results)
        
        print("\n--- Style Transfer Results (Preliminary) ---")
        print(f"Average Normalized Edit Distance to Biblical: {avg_bib:.4f}")
        print(f"Average Normalized Edit Distance to Rabbinic: {avg_rab:.4f}")
        
        if avg_bib < avg_rab:
            print("Result: Modern Hebrew required LESS effort to transform into Biblical Hebrew.")
        else:
            print("Result: Modern Hebrew required LESS effort to transform into Rabbinic Hebrew.")
            
        if is_test_mode:
            print("\n[NOTE] These results are from a dummy model (GPT-Neo-Small) for testing logic.")
            print("They do NOT reflect real linguistic trends. Run without --test for actual results.")

if __name__ == "__main__":
    run_style_transfer()

