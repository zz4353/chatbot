import os
import re
import hashlib
from markitdown import MarkItDown
from jinja2 import Template
from underthesea import word_tokenize


stop_word = set()
with open("res/vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:   
            stop_word.add(word)


def hash_directory(path, exclude_file=None):
    sha256 = hashlib.sha256()

    for root, dirs, files in os.walk(path):
        files.sort()  # Đảm bảo thứ tự cố định
        for file in files:
            if exclude_file and file == exclude_file:
                continue  # Bỏ qua file được loại trừ

            file_path = os.path.join(root, file)
            sha256.update(file_path.encode())  # Thêm đường dẫn

            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)  # Thêm nội dung

    return sha256.hexdigest()

def get_files_in_directory(path):
    files = []
    for f in os.listdir(path):
        path_f = os.path.join(path, f)
        if os.path.isfile(path_f):
            files.append(path_f)
        elif os.path.isdir(path_f):
            files.extend(get_files_in_directory(path_f))

    return files

def convert_to_markdown(path: str) -> str:
    # Khởi tạo converter
    converter = MarkItDown()

    # Chuyển file sang markdown
    result = converter.convert(path)

    return result.markdown


def has_hash_changed(path: str) -> bool:
    # Hash toàn bộ thư mục data
    old_hash = ""
    old_hash_file = os.path.join(path, "hash.hsh")
    if os.path.exists(old_hash_file):
        with open(old_hash_file, "r") as f:
            old_hash = f.read().strip()

    hash = hash_directory(path, "hash.hsh")

    if old_hash == hash:
        return False

    with open(old_hash_file, "w") as f:
        f.write(hash)

    return True


def render_prompt(template_path, docs, question):
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()
    template = Template(template_str)
    return template.render(docs=docs, question=question)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [word.replace(' ', '_') for word in tokens if word not in stop_word]
    return ' '.join(tokens)


if __name__ == "__main__":
    print(preprocess_text("Xin chào, tôi là một trợ lý ảo!"))

