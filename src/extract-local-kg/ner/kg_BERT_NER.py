from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import sys 
from utils import EXTRACT_LOCAL_KG_PATH, DYNAMIC_KGNMT_PATH, get_args

args = get_args()
SPLIT = args.split

input_file_path = DYNAMIC_KGNMT_PATH / f"en-vi-dynamic-kgnmt/kgnmt_data/{SPLIT}.en"
output_file_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/ner.txt"

def perform_ner_with_transformers(input_file_path, output_file_path):
    """
    Performs Named Entity Recognition (NER) on a text file using a Hugging Face Transformer model,
    with a progress indicator showing the current line number being processed.
    Each line in the input file is processed. Named entities are extracted and
    written to the output file, separated by '<k>'.
    Blank lines in the input are preserved as blank lines in the output.
    If no entities are found on a line, a blank line is written.

    Args:
        input_file_path (str): The path to the input text file.
        output_file_path (str): The path to the output file where NER results will be saved.
    """
    try:
        # Load pre-trained tokenizer and model for NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        # Create a NER pipeline, ensuring entities are grouped correctly
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
        print("Hugging Face NER pipeline loaded successfully with 'dslim/bert-base-NER' model.")

    except Exception as e:
        print(f"Error loading Hugging Face model or tokenizer: {e}")
        print("Please ensure you have 'transformers' installed (`pip install transformers`) and an active internet connection to download the model.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        print(f"Processing '{input_file_path}'...")
        line_count = 0
        processed_non_empty_lines = 0

        for line in infile:
            line_count += 1
            original_line_stripped = line.strip() 

            if line_count % 100 == 0:
                print(f"Đang xử lý dòng: {line_count}", end='\r') 
                sys.stdout.flush() 

            if not original_line_stripped:
                outfile.write('\n')
                continue
            
            ner_results = nlp(original_line_stripped)
            
            formatted_entities = []
            for ent in ner_results:
                formatted_entities.append(f"{ent['word']} ({ent['entity_group']})")
            
            if formatted_entities:
                outfile.write('<k>'.join(formatted_entities) + '\n')
            else:
                outfile.write('\n') 
            processed_non_empty_lines += 1

    print(" " * 50, end='\r')
    print(f"Processing complete. Processed {processed_non_empty_lines} non-empty lines out of {line_count} total lines.")
    print(f"Results saved to '{output_file_path}'")


perform_ner_with_transformers(input_file_path, output_file_path)