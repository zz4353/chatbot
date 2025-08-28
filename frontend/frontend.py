import gradio as gr
import requests

def chat_fn(message, history):
    res = requests.post("http://localhost:8000/chat", json={"message": message})
    return res.json()["reply"]

gr.ChatInterface(fn=chat_fn).launch()
