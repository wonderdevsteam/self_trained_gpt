import os

import gradio as gr
import openai
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage

from app.training.train import GPTTrainner

load_dotenv(".env")
persisted_indexes = os.getenv("PERSIST_DIR")

trainner = GPTTrainner(open_ai_key=os.getenv("OPENAI_API_KEY"))

if __name__ == "__main__":
    trainner.build_indexes("docs", persist=persisted_indexes)

