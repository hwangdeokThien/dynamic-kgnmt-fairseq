from langdetect import detect, LangDetectException
import os
from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()
SPLIT = args.split

input_file = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_decoded_triplet.txt" 
output_english_file = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/kg_langdetect.txt"

def read_data(file_path):
    """Reads data from a file and returns a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().replace("<r>", " ").replace("  ", " ") for line in lines]

knowledge_kg = read_data(input_file)

def get_last_processed_index_from_output(output_filepath):
    """
    Đọc file output và trả về chỉ mục của dòng cuối cùng đã được ghi.
    Nếu file không tồn tại hoặc rỗng, trả về 0.
    """
    if not os.path.exists(output_filepath):
        return 0

    last_index = 0
    with open(output_filepath, 'r', encoding='utf-8') as f:
    
        f.seek(0, os.SEEK_END)
        position = f.tell()
        line = ''
        while position >= 0:
            f.seek(position)
            char = f.read(1)
            if char == '\n' and line.strip():
                break
            line = char + line
            position -= 1

        if position < 0 and not line.strip(): 
            return 0

        try:
            index_str = line.split(':', 1)[0].strip()
            last_index = int(index_str)
        except (ValueError, IndexError):
            print(f"Cảnh báo: Dòng cuối cùng của file output '{output_filepath}' không đúng định dạng. Bắt đầu từ đầu.")
            return 0
    return last_index + 1 

start_index = get_last_processed_index_from_output(output_english_file)


english_sentences_count = 0
skipped_sentences_count = 0
total_sentences_in_knowledge_kg = len(knowledge_kg)

with open(output_english_file, 'a', encoding='utf-8') as outfile:

    for original_index in range(start_index, total_sentences_in_knowledge_kg):
        sentence = knowledge_kg[original_index]

        if not sentence.strip(): 
            skipped_sentences_count += 1
            continue

        try:
            if len(sentence.split()) > 2 and detect(sentence) == 'en':
                outfile.write(f"{original_index}: {sentence}\n")
                english_sentences_count += 1
            else:
                skipped_sentences_count += 1
        except LangDetectException:
            skipped_sentences_count += 1
            continue
        except Exception as e:
            print(f"Lỗi không xác định khi xử lý câu {original_index}: '{sentence}'. Lỗi: {e}")
            skipped_sentences_count += 1
            continue
        
        if (original_index + 1) % 10000 == 0 or (original_index + 1) == total_sentences_in_knowledge_kg:
            progress = ((original_index + 1) / total_sentences_in_knowledge_kg) * 100
            print(f"Đang xử lý câu thứ {original_index + 1}/{total_sentences_in_knowledge_kg} ({progress:.2f}%). Đã ghi {english_sentences_count} câu tiếng Anh.")

print(f"\n--- Quá trình lọc hoàn tất ---")
print(f"Tổng số câu tiếng Anh đã lọc và ghi mới trong lần chạy này: {english_sentences_count}")
print(f"Tổng số câu bị bỏ qua (không phải tiếng Anh, quá ngắn hoặc lỗi) trong lần chạy này: {skipped_sentences_count}")
print(f"File kết quả được lưu tại: {os.path.abspath(output_english_file)}")
print(f"Lần chạy tiếp theo sẽ bắt đầu từ chỉ mục: {get_last_processed_index_from_output(output_english_file)}")