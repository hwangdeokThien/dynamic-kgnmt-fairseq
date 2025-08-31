import os 
from utils import EXTRACT_LOCAL_KG_PATH

triplet_file = "wikidata5m_all_triplet.txt"     #file wikidata5m_all_triplet
entity_file = "wikidata5m_entity.txt"           #file wikidata5m_entity
relation_file = "wikidata5m_relation.txt"       #file wikidata5m_relation
output_file = "wikidata5m_decoded_triplet.txt"  #file output

if os.path.exists(EXTRACT_LOCAL_KG_PATH / f"data-kg/{output_file}"):
    exit()

def decode_wikidata_triplets(triplet_file_path, entity_file_path, relation_file_path, output_file_path):
    """
    Giải mã các triple Wikidata từ ID sang nhãn người đọc được 
    Args:
        triplet_file_path (str): Đường dẫn đến file chứa các triple (e.g., wikidata5m_all_triplet.txt).
        entity_file_path (str): Đường dẫn đến file ánh xạ thực thể (e.g., wikidata5m_entity.txt).
        relation_file_path (str): Đường dẫn đến file ánh xạ quan hệ (e.g., wikidata5m_relation.txt).
        output_file_path (str): Đường dẫn đến file đầu ra chứa các triple đã giải mã.
    """

    print("Đang đọc file thực thể...")
    entity_map = {}
    with open(entity_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entity_id = parts[0]
                entity_label = parts[1].split("\\t")[0].strip()
                entity_map[entity_id] = entity_label
    print(f"Đã đọc {len(entity_map)} thực thể.")

    print("Đang đọc file quan hệ...")
    relation_map = {}
    with open(relation_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                relation_id = parts[0]
                relation_label = parts[1].split("\\t")[0].strip()
                relation_map[relation_id] = relation_label
    print(f"Đã đọc {len(relation_map)} quan hệ.")

    print("Đang giải mã các triple...")
    decoded_triplets_count = 0
    with open(triplet_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            s_id, p_id, o_id = line.strip().split("\t")

            s_label = entity_map.get(s_id, s_id) 
            p_label = relation_map.get(p_id, p_id)
            o_label = entity_map.get(o_id, o_id) 

            outfile.write(f"{s_label}<r>{p_label}<r>{o_label}\n")
            decoded_triplets_count += 1
            if decoded_triplets_count % 100000 == 0:
                print(f"Đã giải mã {decoded_triplets_count} triple...")

    print(f"Hoàn tất giải mã. Đã ghi {decoded_triplets_count} triple vào {output_file_path}")

 

decode_wikidata_triplets(triplet_file, entity_file, relation_file, output_file)