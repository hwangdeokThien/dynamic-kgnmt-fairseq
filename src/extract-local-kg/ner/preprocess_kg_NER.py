import re
import os
import sys 
from utils import EXTRACT_LOCAL_KG_PATH, SPLIT

input_file_path = EXTRACT_LOCAL_KG_PATH / f"data-kg/{SPLIT}/ner.txt"
output_file_path = EXTRACT_LOCAL_KG_PATH / "data-kg/{SPLIT}/ner_cleaned.txt"

def clean_text_segment(text_segment):
    """
    Loại bỏ các từ trong đoạn text chứa ký tự đặc biệt VÀ loại bỏ nội dung trong ngoặc đơn '()'.
    Các ký tự đặc biệt được định nghĩa là bất kỳ ký tự nào KHÔNG phải là chữ cái (tiếng Việt hoặc tiếng Anh),
    số, hoặc dấu gạch nối (hyphen).
    """

    text_without_parentheses = re.sub(r'\s*\([^)]*\)', '', text_segment).strip()

    words = re.split(r'\s+', text_without_parentheses)

    cleaned_words = []

    invalid_char_pattern = re.compile(r'[^a-zA-Z0-9\-_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚỦỤƯỪỬỰÝỲỸỶỶĐàáâãèéêìíòóôõùúũủụưừửựýỳỹỷỷđ]')

    for word in words:
        if word and not invalid_char_pattern.search(word):

            cleaned_words.append(word)

    return ' '.join(cleaned_words)

def process_file_and_clean_text(input_file_path, output_file_path, separator='<k>'):
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục output: {output_dir}")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            print(f"Đang xử lý file: '{input_file_path}'...")
            line_count = 0

            for line in infile:
                line_count += 1
                original_line_stripped = line.strip()

                if line_count % 100 == 0:
                    print(f"Đang xử lý dòng: {line_count}", end='\r')
                    sys.stdout.flush()

                if not original_line_stripped:
                    outfile.write('\n')
                    continue

                text_segments = original_line_stripped.split(separator)

                cleaned_segments = []
                for segment in text_segments:
                    cleaned_segment = clean_text_segment(segment)
                    cleaned_segments.append(cleaned_segment)

                outfile.write(separator.join(cleaned_segments) + '\n')
            print(" " * 50, end='\r') 
            print(f"Xử lý hoàn tất. Tổng cộng {line_count} dòng đã được xử lý.")
            print(f"Kết quả đã được lưu vào '{output_file_path}'")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file input tại đường dẫn '{input_file_path}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý file: {e}")


process_file_and_clean_text(input_file_path, output_file_path)