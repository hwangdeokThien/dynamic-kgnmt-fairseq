from collections import Counter
import re
import random
import os
from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()
SPLIT = args.split

file_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/triple_langdetect.txt"
output_dir = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/extract_popular_rare_random/"

def read_data(file_path):
    """Reads data from a file and returns a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().replace("<r>", " ").replace("  ", " ") for line in lines]

raw_sentences_list = read_data(file_path)

knowledge_kg = []
for i, sentence_text in enumerate(raw_sentences_list):
    if sentence_text: 
        knowledge_kg.append((sentence_text, f"line_{i}")) 


print(f"Tổng số câu trong dữ liệu gốc (kèm thông tin nguồn): {len(knowledge_kg)}")

word_frequencies = Counter()
for sentence_text, _ in knowledge_kg: 
    words = re.findall(r'\b\w+\b', sentence_text.lower())
    word_frequencies.update(words)

word_rarity_scores = {word: 1 / (freq + 1) for word, freq in word_frequencies.items()}

sentence_rarity_scores = []

for i, (sentence_text, source_identifier) in enumerate(knowledge_kg):
    words = re.findall(r'\b\w+\b', sentence_text.lower())
    sentence_score = sum(word_rarity_scores.get(word, 0) for word in words)
    sentence_rarity_scores.append((sentence_score, sentence_text, i, source_identifier))


sentence_rarity_scores.sort(key=lambda x: x[0], reverse=True)

num_sentences_to_take = 100000

num_rare_sentences = min(num_sentences_to_take, len(sentence_rarity_scores))
rare_sentences = sentence_rarity_scores[:num_rare_sentences]
print(f"\nSố lượng câu hiếm nhất được chọn: {len(rare_sentences)}")

popular_sentences = []
rare_original_indices = {item[2] for item in rare_sentences}

for item in reversed(sentence_rarity_scores):
    if item[2] not in rare_original_indices: 
        popular_sentences.append(item)
    if len(popular_sentences) >= num_sentences_to_take:
        break
popular_sentences = popular_sentences[:num_sentences_to_take]

print(f"Số lượng câu phổ biến nhất được chọn: {len(popular_sentences)}")

selected_indices = rare_original_indices.union({item[2] for item in popular_sentences})
remaining_sentences = [item for item in sentence_rarity_scores if item[2] not in selected_indices]
print(f"Số lượng câu còn lại: {len(remaining_sentences)}")

num_sets = 10
set_size = 100000
sampled_sets = []

if len(remaining_sentences) >= set_size:
    print(f"\nBắt đầu tạo {num_sets} bộ mẫu ngẫu nhiên (mỗi bộ {set_size} câu)...")
    for i in range(num_sets):
        random.shuffle(remaining_sentences) 
        sampled_set = remaining_sentences[:set_size] 
        sampled_sets.append(sampled_set)
        print(f"   > Đã tạo bộ mẫu ngẫu nhiên thứ {i+1}: {len(sampled_set)} câu.")
else:
    print(f"\nKhông đủ câu còn lại ({len(remaining_sentences)}) để tạo {num_sets} bộ, mỗi bộ {set_size} câu.")
    if remaining_sentences:
        random.shuffle(remaining_sentences)
        sampled_sets.append(remaining_sentences)
        print(f"   > Đã tạo 1 bộ mẫu ngẫu nhiên với tất cả {len(remaining_sentences)} câu còn lại.")


def write_sentences_to_file(data_list, file_path, description):
    """
    Writes a list of sentences (with their score, original index, and source identifier) to a file.
    Each item in data_list is expected to be a tuple:
    (score, sentence_text, original_index, source_identifier)
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Original_Global_Index\tSource_Identifier\tScore\tSentence_Text\n")
        for score, sentence_text, original_index, source_identifier in data_list:
            f.write(f"{original_index}\t{source_identifier}\t{score:.4f}\t{sentence_text}\n")
    print(f"\nĐã xuất {len(data_list)} câu {description} vào file '{file_path}'.")

os.makedirs(output_dir, exist_ok=True) 
write_sentences_to_file(rare_sentences, os.path.join(output_dir, "rare_sentences_100k.txt"), "hiếm nhất")
write_sentences_to_file(popular_sentences, os.path.join(output_dir, "popular_sentences_100k.txt"), "phổ biến nhất")
for i, sampled_set in enumerate(sampled_sets):
    file_name = f"random_sampled_set_{i+1}_100k.txt"
    write_sentences_to_file(sampled_set, os.path.join(output_dir, file_name), f"mẫu ngẫu nhiên thứ {i+1}")

print("\nQuá trình tạo và xuất các tập dữ liệu đã hoàn tất.")
print(f"Các file đã được lưu trong thư mục: {output_dir}")