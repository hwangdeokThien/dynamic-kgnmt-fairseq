import os
import sys 
from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()
SPLIT = args.split

WIKIDATA5M_TRIPLET_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_all_triplet.txt" # path to file wikidata5m_all_triplet
INPUT_ENTITY_IDS_FILE = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/entity_ids.txt"
OUTPUT_TRIPLES_FILE = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/extracted_triples.txt"

SEPARATOR = '<k>'

def extract_unique_entity_ids(input_file_path, separator=SEPARATOR):
    """
    Trích xuất tất cả các Entity ID duy nhất từ file input.
    """
    unique_ids = set()
    print(f"Đang trích xuất các Entity ID duy nhất từ '{input_file_path}'...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000 == 0:
                    print(f"Đọc dòng ID: {line_num}", end='\r')
                    sys.stdout.flush()

                line = line.strip()
                if not line:
                    continue

                ids_on_line = line.split(separator)
                for _id in ids_on_line:
                    if _id.startswith('Q') and _id[1:].isdigit():
                        unique_ids.add(_id)
        print(f"\nĐã trích xuất {len(unique_ids)} Entity ID duy nhất.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file Entity ID tại '{input_file_path}'.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file Entity ID: {e}")
        return None
    return unique_ids

def load_relevant_triples_into_memory(entity_ids_to_find, triplet_file_path):
    """
    Tải các triple có subject ID nằm trong entity_ids_to_find vào bộ nhớ.
    Trả về một dictionary: {subject_id: [triple1_str, triple2_str, ...]}
    """
    if not entity_ids_to_find:
        print("Không có Entity ID nào để tải triple liên quan.")
        return {}

    relevant_triples = {}
    print(f"Đang quét '{triplet_file_path}' để tải các triple liên quan vào bộ nhớ...")
    found_triples_count = 0

    try:
        with open(triplet_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                if line_num % 1000000 == 0: 
                    print(f"Đang quét triple: {line_num} (Đã tải: {found_triples_count})", end='\r')
                    sys.stdout.flush()

                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    subject_id = parts[0]
                    if subject_id in entity_ids_to_find:
                        if subject_id not in relevant_triples:
                            relevant_triples[subject_id] = []
                        relevant_triples[subject_id].append(line.strip()) 
                        found_triples_count += 1

        print(" " * 80, end='\r') 
        print(f"Hoàn tất tải triple liên quan. Tổng số triple đã tải: {found_triples_count}")
        print(f"Tổng số Entity ID có triple liên quan: {len(relevant_triples)}")
        return relevant_triples

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file triple tại '{triplet_file_path}'. Vui lòng kiểm tra đường dẫn.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải triple liên quan: {e}")
        return None

def format_triples_to_output(input_entity_ids_file, loaded_triples_map, output_file_path, separator=SEPARATOR):
    """
    Đọc file input chứa các Entity ID, tìm các triple tương ứng từ loaded_triples_map,
    và định dạng output theo yêu cầu.
    """
    if not loaded_triples_map:
        print("Không có triple nào được tải. Không thể định dạng output.")
        return

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output: {output_dir}")

    print(f"Đang định dạng output cho file '{input_entity_ids_file}'...")

    try:
        with open(input_entity_ids_file, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            line_count = 0
            for line in infile:
                line_count += 1
                original_line_stripped = line.strip()

                if line_count % 100 == 0:
                    print(f"Đang định dạng dòng input: {line_count}", end='\r')
                    sys.stdout.flush()

                if not original_line_stripped:
                    outfile.write('\n')
                    continue

                entity_ids_on_line = original_line_stripped.split(separator)

                all_triples_for_line = []
                for entity_id in entity_ids_on_line:
                    if entity_id in loaded_triples_map:
                        all_triples_for_line.extend(loaded_triples_map[entity_id])

                if all_triples_for_line:
                    outfile.write(separator.join(all_triples_for_line) + '\n')
                else:
                    outfile.write('\n')

            print(" " * 80, end='\r') 
            print(f"Hoàn tất định dạng output. Kết quả đã lưu vào '{output_file_path}'")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file Entity ID input tại đường dẫn '{input_entity_ids_file}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình định dạng output: {e}")


target_entity_ids = extract_unique_entity_ids(INPUT_ENTITY_IDS_FILE)

loaded_triples_map = {}

if target_entity_ids:
    loaded_triples_map = load_relevant_triples_into_memory(target_entity_ids, WIKIDATA5M_TRIPLET_PATH)

if loaded_triples_map is not None: 
    format_triples_to_output(INPUT_ENTITY_IDS_FILE, loaded_triples_map, OUTPUT_TRIPLES_FILE)