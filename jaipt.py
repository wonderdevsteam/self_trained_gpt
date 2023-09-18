import os
import sys

import gradio as gr
import openai
from dotenv import load_dotenv
from llama_index import (
    GPTListIndex,StorageContext,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,load_index_from_storage
)
from langchain.chat_models import ChatOpenAI

load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        chunk_size_limit=chunk_size_limit,
    )

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.storage_context.persist(persist_dir="trained_indexes")

    return index


def chatbot(input_text):
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="trained_indexes")).as_query_engine()
    response = index.query(input_text)
    return response.response

if __name__ == "__main__":
    print("Start chatting with the bot (type 'quit' to stop)!")
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
        outputs="text",
        title="Mea testa",
    )

    # index = construct_index("docs")
    iface.launch(share=True)
