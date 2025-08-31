import os
import sys
import re
from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()
SPLIT = args.splits

INPUT_FORMATTED_TRIPLES_FILE = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/extracted_triples.txt"     # file input
WIKIDATA5M_ENTITY_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_entity.txt"                   # file wikidata5m_entity
WIKIDATA5M_RELATION_PATH = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_relation.txt"               # file wikidata5m_relation
OUTPUT_DECODED_TRIPLES_FILE = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/decoded_triples.txt"        # file output

TRIPLE_SEPARATOR = '<k>'
TRIPLE_PART_SEPARATOR_INPUT = '\t' 
TRIPLE_PART_SEPARATOR_OUTPUT = '<r>' 

entity_id_to_label = {}
relation_id_to_label = {}

def load_id_to_label_map(file_path, id_prefix):
    """
    Tải ánh xạ từ ID sang nhãn từ file entity hoặc relation.
    Chỉ lấy nhãn đầu tiên ứng với mỗi ID.
    """
    mapping = {}
    print(f"Đang tải ánh xạ ID -> Nhãn từ '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000000 == 0:
                    print(f"Đọc dòng map {id_prefix}: {line_num}", end='\r')
                    sys.stdout.flush()
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    _id = parts[0]
                    label = parts[1]
                    if _id.startswith(id_prefix) and _id not in mapping:
                        mapping[_id] = label
        print(f"\nHoàn tất tải ánh xạ từ '{file_path}'. Tổng số {len(mapping)} mục.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file '{file_path}': {e}")
        return None
    return mapping

entity_id_to_label = load_id_to_label_map(WIKIDATA5M_ENTITY_PATH, 'Q')
if entity_id_to_label is None:
    print("Không thể tiếp tục do lỗi tải ánh xạ Entity ID.")
    sys.exit(1)
relation_id_to_label = load_id_to_label_map(WIKIDATA5M_RELATION_PATH, 'P')
if relation_id_to_label is None:
    print("Không thể tiếp tục do lỗi tải ánh xạ Relation ID.")
    sys.exit(1)

def decode_entity_or_literal(id_or_literal, entity_map):
    """
    Giải mã Entity ID thành nhãn, hoặc trả về literal nếu không phải ID.
    """
    if id_or_literal.startswith('Q') and id_or_literal[1:].isdigit():
        return entity_map.get(id_or_literal, id_or_literal) 
    return id_or_literal 

def decode_triples_file(input_file_path, output_file_path, entity_map, relation_map,
                        triple_separator=TRIPLE_SEPARATOR,
                        triple_part_separator_input=TRIPLE_PART_SEPARATOR_INPUT,
                        triple_part_separator_output=TRIPLE_PART_SEPARATOR_OUTPUT):
    """
    Đọc file chứa các triple ID, giải mã chúng và ghi vào file output.
    Sử dụng triple_part_separator_output để phân tách Subject, Predicate, Object.
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output: {output_dir}")

    print(f"\nĐang giải mã các triple trong '{input_file_path}'...")
    processed_lines = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile):
                processed_lines += 1
                original_line_stripped = line.strip()

                if processed_lines % 100 == 0:
                    print(f"Đang giải mã dòng: {processed_lines}", end='\r')
                    sys.stdout.flush()

                if not original_line_stripped:
                    outfile.write('\n') 
                    continue

                id_triples = original_line_stripped.split(triple_separator)

                decoded_triples_for_line = []
                for id_triple_str in id_triples:
                    parts = id_triple_str.split(triple_part_separator_input)
                    if len(parts) >= 3:
                        s_id = parts[0].strip()
                        p_id = parts[1].strip()
                        o_id_or_literal = parts[2].strip()

                        decoded_s = decode_entity_or_literal(s_id, entity_map)

                        decoded_p = relation_map.get(p_id, p_id) 

                        decoded_o = decode_entity_or_literal(o_id_or_literal, entity_map)

                        decoded_triples_for_line.append(f"{decoded_s}{triple_part_separator_output}{decoded_p}{triple_part_separator_output}{decoded_o}")

                if decoded_triples_for_line:
                    outfile.write(triple_separator.join(decoded_triples_for_line) + '\n')
                else:
                    outfile.write('\n') 

            print(" " * 80, end='\r') 
            print(f"Hoàn tất giải mã triple. Tổng số dòng đã xử lý: {processed_lines}")
            print(f"Kết quả đã được lưu vào '{output_file_path}'")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file input tại đường dẫn '{input_file_path}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình giải mã triple: {e}")


decode_triples_file(INPUT_FORMATTED_TRIPLES_FILE, OUTPUT_DECODED_TRIPLES_FILE,
                    entity_id_to_label, relation_id_to_label,
                    TRIPLE_SEPARATOR, TRIPLE_PART_SEPARATOR_INPUT, TRIPLE_PART_SEPARATOR_OUTPUT)