import os
import re
from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()
SPLIT = args.split

WIKIDATA5M_ENTITY_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_entity.txt" # path to file wikidata5m_entity 
INPUT_NER_FILE_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/ner_cleaned.txt"
OUTPUT_ENTITY_ID_FILE_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/entity_ids.txt"

entity_label_to_id = {}
entity_alias_to_id = {} 

def load_wikidata5m_entities(file_path):
    """
    Tải thông tin thực thể từ file wikidata5m_entity và xây dựng ánh xạ.
    """
    print(f"Đang tải và xây dựng ánh xạ từ '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:
                    print(f"Đang đọc dòng Wikidata: {line_num}", end='\r')
                    import sys
                    sys.stdout.flush()

                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    label = parts[1].lower() 

                    if label not in entity_label_to_id:
                        entity_label_to_id[label] = entity_id

               
                    if len(parts) >= 4 and parts[3]: 
                        aliases = parts[3].split(';')
                        for alias in aliases:
                            alias_lower = alias.strip().lower()
                            if alias_lower and alias_lower not in entity_alias_to_id:
                                entity_alias_to_id[alias_lower] = entity_id
            print("\nHoàn thành tải Wikidata5M entities.")
            print(f"Đã tải {len(entity_label_to_id)} nhãn chính và {len(entity_alias_to_id)} bí danh.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file wikidata5m_entity tại '{file_path}'. Vui lòng kiểm tra đường dẫn.")
        return False
    except Exception as e:
        print(f"Lỗi khi đọc file wikidata5m_entity: {e}")
        return False
    return True

#-------------------------------------------------------------------------------------------------------------------
def find_entity_id(entity_text):

    cleaned_text = re.sub(r'\s*\([^)]+\)$', '', entity_text).strip().lower()

    if not cleaned_text:
        return None 

    if cleaned_text in entity_label_to_id:
        return entity_label_to_id[cleaned_text]

    if cleaned_text in entity_alias_to_id:
        return entity_alias_to_id[cleaned_text]

    return None 

#-------------------------------------------------------------------------------------------------------------------
def process_ner_file_to_entity_ids(input_file_path, output_file_path, separator='<k>'):
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output: {output_dir}")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            print(f"\nĐang xử lý file input: '{input_file_path}' để tìm Entity ID...")
            line_count = 0

            for line in infile:
                line_count += 1
                original_line_stripped = line.strip()

                if line_count % 100 == 0: 
                    print(f"Đang xử lý dòng: {line_count}", end='\r')
                    import sys
                    sys.stdout.flush()

                if not original_line_stripped:
               
                    outfile.write('\n')
                    continue

                entity_segments = original_line_stripped.split(separator)

                linked_ids = []
                for entity_text_with_type in entity_segments:
                    entity_id = find_entity_id(entity_text_with_type)
                    if entity_id:
                        linked_ids.append(entity_id)

                if linked_ids:
                    outfile.write(separator.join(linked_ids) + '\n')
                else:
                    outfile.write('\n')

            print(" " * 50, end='\r') 
            print(f"Hoàn tất liên kết thực thể. Tổng cộng {line_count} dòng đã được xử lý.")
            print(f"Kết quả Entity ID đã được lưu vào '{output_file_path}'")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file input tại đường dẫn '{input_file_path}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý file: {e}")

# --- Cấu hình đường dẫn ---
if not load_wikidata5m_entities(WIKIDATA5M_ENTITY_PATH):
    print("Không thể tiếp tục xử lý do lỗi tải dữ liệu Wikidata5M.")
    exit() 

process_ner_file_to_entity_ids(INPUT_NER_FILE_PATH, OUTPUT_ENTITY_ID_FILE_PATH)
