from utils import EXTRACT_LOCAL_KG_PATH, get_args

args = get_args()   
SPLIT = args.split

file_1_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/kg_langdetect.txt"
file_2_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/wikidata5m_decoded_triplet.txt"
output_filename = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/triple_langdetect.txt" 

parsed_file_1 = {}
try:
    with open(file_1_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() 
            if line: 
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    try:
                        index = int(parts[0])
                        content = parts[1]
                        parsed_file_1[index] = content
                    except ValueError:
                        print(f"Bỏ qua dòng không đúng định dạng index: {line}")
                        continue
except FileNotFoundError:
    print(f"Lỗi: File không tìm thấy tại đường dẫn: {file_1_path}")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc File 1: {e}")
    exit()

print(f"Đã đọc {len(parsed_file_1)} mục từ File 1.")


file_2_list = []
try:
    with open(file_2_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_2_list.append(line.strip())
except FileNotFoundError:
    print(f"Lỗi: File không tìm thấy tại đường dẫn: {file_2_path}")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc File 2: {e}")
    exit()

print(f"Đã đọc {len(file_2_list)} mục từ File 2.")




filtered_content_from_file_2 = []
count_filtered = 0

for index in indexes_to_filter:
    if index < len(file_2_list):
        filtered_content_from_file_2.append(file_2_list[index])
        count_filtered += 1


print(f"Đã lọc được {count_filtered} mục từ File 2.")


try:
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for item in filtered_content_from_file_2:
            outfile.write(item + '\n')
    print(f"Nội dung đã lọc đã được ghi vào file: {output_filename}")
except Exception as e:
    print(f"Lỗi khi ghi kết quả ra file: {e}")