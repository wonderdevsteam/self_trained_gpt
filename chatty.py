import os

import gradio as gr
import openai
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage

load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
persisted_indexes = os.getenv("PERSIST_DIR")


def chatbot(input_text):
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persisted_indexes)
    ).as_query_engine()

    response = index.query(input_text)

    return response.response


if __name__ == "__main__":
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
        outputs="text",
        title="Chatty",
    )

    iface.launch(share=True)
