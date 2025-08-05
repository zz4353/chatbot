from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import load_file_as_markdown

def load_documents_from_path(path):
    chunk_contents = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=512, chunk_overlap=126
            )

    # Xử lý file .docx, txt và .pdf
    if path.endswith(".docx") or path.endswith(".pdf") or path.endswith(".txt"):
        print(f"Processing file: {path}")
        chunk_contents = process_file(path, text_splitter)
    else:
        pass

    return chunk_contents

def process_file(path, text_splitter):
    """ docx, pdf, txt"""
    file_name = path.split("/")[-1]
    markdown_text = load_file_as_markdown(path)
    chunks = text_splitter.split_text(markdown_text)
    chunk_contents = ["Nguồn: " + file_name + '\n' + chunk.replace("\n", " ") for chunk in chunks if len(chunk) >= 40]
    return chunk_contents