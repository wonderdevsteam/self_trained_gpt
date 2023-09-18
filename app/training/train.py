from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
)


class GPTTrainner:
    MAX_INTPUT_SIZE = 4096
    NUM_OUTPUTS = 512
    CHUNCK_SIZE_LIMIT = 600
    OVERLAP_RATIO = 0.2

    def __init__(
        self, open_ai_key: str | None = None, gpt_model: str = "gpt-3.5-turbo"
    ):
        self.__model = gpt_model
        self.__secret = open_ai_key

    def build_indexes(self, directory_path: str, persist: str | None = None):
        prompt_helper = PromptHelper(
            self.MAX_INTPUT_SIZE,
            self.NUM_OUTPUTS,
            self.OVERLAP_RATIO,
            chunk_size_limit=self.CHUNCK_SIZE_LIMIT,
        )

        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                temperature=0.7,
                model_name=self.__model,
                max_tokens=self.NUM_OUTPUTS,
                openai_api_key=self.__secret,
            )
        )

        documents = SimpleDirectoryReader(directory_path).load_data()

        index = GPTVectorStoreIndex(
            documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )

        if persist:
            index.storage_context.persist(persist_dir=persist)

        return index
