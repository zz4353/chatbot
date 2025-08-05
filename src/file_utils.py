from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_files_in_directory, convert_to_markdown
from langchain.schema import Document
import json

def load_documents_from_path(path):
    chunk_contents = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=512, chunk_overlap=126
            )

    file_name = path.split("/")[-1]

    # Xử lý file .docx, .txt và .pdf
    if path.endswith(".docx") or path.endswith(".txt") or path.endswith(".pdf"):
        print(f"Processing file: {path}")
        markdown_text = convert_to_markdown(path)
        docs = [Document(page_content=markdown_text)]
        chunks = text_splitter.transform_documents(docs)
        chunk_contents = ["Nguồn: " + file_name + '\n' + doc.page_content.replace("\n", " ") for doc in chunks if len(doc.page_content) >= 40]

    # Xử lý file .json
    elif path.endswith(".json"):
        print(f"Processing file: {path}")
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        for entry in data:
            content = entry.get("content", "")
            metadata = entry.get("metadata", "")

            if not content.strip():
                continue

            docs = [Document(page_content=content)]
            chunks = text_splitter.transform_documents(docs)

            # Gắn metadata vào mỗi chunk
            for doc in chunks:
                if len(doc.page_content) >= 40:
                    chunk_contents.append(f"[{metadata}]\n{doc.page_content}")

    else:
        pass

    return chunk_contents
