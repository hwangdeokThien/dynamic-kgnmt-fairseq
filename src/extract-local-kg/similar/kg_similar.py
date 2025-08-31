import numpy as np
import pandas as pd
import json
import io
import torch
import time
from heapq import heappush, heappop
from sentence_transformers import SentenceTransformer
import os
from utils import EXTRACT_LOCAL_KG_PATH, DYNAMIC_KGNMT_PATH, get_args

args = get_args()
SPLIT = args.split

data_dir = DYNAMIC_KGNMT_PATH / f'en-vi-dynamic-kgnmt/kgnmt_data/{SPLIT}.en' # thư mục data câu nguồn
kg_trans_dir = EXTRACT_LOCAL_KG_PATH / f'data-kg/kg_trans.txt' # thư mục data kg_translation
kg_dir = EXTRACT_LOCAL_KG_PATH / f'data-kg/{SPLIT}/extract_popular_rare_random/popular_sentences.txt' # thư mục data kg tren ngon ngu nguon
output_file = EXTRACT_LOCAL_KG_PATH / f'data-kg/{SPLIT}/all.kg' # file output

def process_knowledge_graph_data(data_dir, kg_trans_dir, kg_dir, output_file):
    """
    Processes knowledge graph data and finds top-k similar KG sentences
    for each source English sentence.

    Args:
        data_dir (str): Directory of the source English sentences (split.en).
        kg_trans_dir (str): Directory of the translated KG data (kg_trans.txt).
        kg_dir (str): Directory of the source language KG data (popular_sentences.txt).
        output_file (str): Path to the output file to save the results (all.kg).
    """

    # --- Load and Prepare Data ---
    with open(data_dir, 'r', encoding='utf-8') as f:
        data_list = [line.strip() for line in f]
    df_data = pd.DataFrame(data_list, columns=['en_sentence'])

    df_kg_en = pd.read_csv(
        kg_dir,
        sep='<r>',
        header=None,
        names=['entity1', 'relation', 'entity2'],
        engine='python'
    )

    df_kg_trans = pd.read_csv(
        kg_trans_dir,
        sep='<r>',
        header=None,
        names=['entity1', 'relation', 'entity2'],
        engine='python'
    )

    df_combined = pd.concat([df_kg_en, df_kg_trans], ignore_index=True)
    df_kg = df_combined.sample(frac=1).reset_index(drop=True)
    df_kg['kg_text'] = df_kg['entity1'] + ' ' + df_kg['relation'] + ' ' + df_kg['entity2']
    df_kg.loc[df_kg['relation'] == ' <t> ', 'kg_text'] = df_kg['entity1']

    # --- Create Output Directory ---
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Initialize SentenceTransformer Model ---
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    print(f"\nUsing SentenceTransformer model: all-MiniLM-L6-v2 on device: {model.device}")

    all_source_sentences = df_data['en_sentence'].tolist()

    if not all_source_sentences:
        print("No source sentences found in df_data. Output file will be empty. Exiting.")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            pass
        return 

    print(f"\nEncoding {len(all_source_sentences)} source sentences...")
    source_embeddings = model.encode(all_source_sentences, convert_to_tensor=True, show_progress_bar=True)
    print(f"Source embeddings shape: {source_embeddings.shape}")

    if source_embeddings.shape[0] != len(all_source_sentences):
        print(f"Warning: SentenceTransformer encoded {source_embeddings.shape[0]} embeddings, but {len(all_source_sentences)} sentences were provided. Missing embeddings will result in empty top-k lists for those source sentences. This typically happens if source sentences were empty strings and the model skips them.")

    # --- Process KG Sentences in Batches ---
    batch_size = 5000
    top_k = 100
    top_k_heap = {i: [] for i in range(len(all_source_sentences))}

    merged_kg_sentences_for_embedding = df_kg['kg_text'].tolist()
    print(f"Total merged_kg sentences to process: {len(merged_kg_sentences_for_embedding)}")

    total_batches = (len(merged_kg_sentences_for_embedding) // batch_size) + (1 if len(merged_kg_sentences_for_embedding) % batch_size != 0 else 0)

    for batch_start in range(0, len(merged_kg_sentences_for_embedding), batch_size):
        start_time = time.time()
        batch_end = min(batch_start + batch_size, len(merged_kg_sentences_for_embedding))
        batch_sentences = merged_kg_sentences_for_embedding[batch_start:batch_end]
        print(f"Processing batch {batch_start // batch_size + 1}/{total_batches}", end=' - ')

        batch_embeddings = model.encode(batch_sentences, convert_to_tensor=True)

        if batch_embeddings.numel() == 0:
            print("Skipping empty batch (all KG sentences were empty or resulted in no embeddings).")
            continue

        similarities = torch.mm(source_embeddings, batch_embeddings.T)

        for src_idx in range(len(all_source_sentences)):
            batch_scores = similarities[src_idx].cpu().tolist()
            for batch_idx, score in enumerate(batch_scores):
                if len(top_k_heap[src_idx]) < top_k:
                    heappush(top_k_heap[src_idx], (score, batch_start + batch_idx))
                elif score > top_k_heap[src_idx][0][0]:
                    heappop(top_k_heap[src_idx])
                    heappush(top_k_heap[src_idx], (score, batch_start + batch_idx))

        end_time = time.time()
        print(f"GPU-accelerated time: {end_time - start_time:.2f} seconds")

    # --- Save Results ---
    print(f"\nSaving top {top_k} results to {output_file}, maintaining original line alignment...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, query_sentence_original in enumerate(df_data['en_sentence']):
            if query_sentence_original.strip() == "":
                outfile.write('\n')
            else:
                sorted_results = sorted(top_k_heap[idx], reverse=True)

                kg_sentences_formatted = []
                for score, kg_original_idx in sorted_results:
                    kg_row = df_kg.iloc[kg_original_idx]
                    formatted_kg_string = f"{kg_row['entity1']} <r> {kg_row['relation']} <r> {kg_row['entity2']}"
                    kg_sentences_formatted.append(formatted_kg_string)

                if kg_sentences_formatted:
                    outfile.write(f"{'<k>'.join(kg_sentences_formatted)}\n")
                else:
                    outfile.write('\n')

    print("Processing complete and results saved.")


process_knowledge_graph_data(data_dir, kg_trans_dir, kg_dir, output_file)