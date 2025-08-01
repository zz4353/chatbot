1. Cài docker, sau đó chạy script sau để cài đặt qdrant

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
  
2. Cài Ollama sau đó cài qwen3 4b (hiện tại đang dùng bản này)

ollama serve &
ollama run qwen3:4b

3. Các thư viện còn lại, xem trong phần import, nhờ chatGPT chỉ ra để cài.

4. Chạy: Chạy file main.py là đủ. Tương tác qua terminal
