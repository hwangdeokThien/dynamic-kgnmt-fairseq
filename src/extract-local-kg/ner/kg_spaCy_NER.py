import spacy
from utils import EXTRACT_LOCAL_KG_PATH, DYNAMIC_KGNMT_PATH, get_args

args = get_args()
SPLIT = args.split

input_file_path = DYNAMIC_KGNMT_PATH / f"en-vi-dynamic-kgnmt/kgnmt_data/{SPLIT}.en"
output_file_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/ner.txt"

def perform_ner_on_file(input_file_path, output_file_path):
    """
    Performs Named Entity Recognition (NER) on a text file.
    Each line in the input file is processed, and named entities are
    extracted and written to the output file, separated by '<k>'.
    Blank lines in the input are preserved as blank lines in the output.

    Args:
        input_file_path (str): The path to the input text file.
        output_file_path (str): The path to the output file where NER results will be saved.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"SpaCy model '{nlp.meta['name']}' loaded successfully.")

    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Please download it by running:")
        print("python -m spacy download en_core_web_sm")
        return

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        print(f"Processing '{input_file_path}'...")
        line_count = 0
        processed_line_count = 0

        for line in infile:
            line_count += 1
            line = line.strip() 

            if not line:
                outfile.write('\n')
                continue

            doc = nlp(line)
            entities = [f"{ent.text}" for ent in doc.ents]

            if entities:
                outfile.write('<k>'.join(entities) + '\n')
            else:
                outfile.write('\n')
            processed_line_count += 1

    print(f"Processing complete. Processed {processed_line_count} non-empty lines out of {line_count} total lines.")
    print(f"Results saved to '{output_file_path}'")


perform_ner_on_file(input_file_path, output_file_path)