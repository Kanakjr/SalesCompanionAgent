# -*- coding: utf-8 -*-
import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import AzureOpenAI, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="./data/.langchain.db"))

def load_openai_embeddings():
    if os.environ[f"GENAI_TYPE"] == "OpenAI" or os.environ[f"GENAI_TYPE"] == "ChatOpenAI":
        return OpenAIEmbeddings()
    elif os.environ[f"GENAI_TYPE"] == "AzureOpenAI" or os.environ[f"GENAI_TYPE"] == "AzureOpenAIChat":
        return AzureOpenAIEmbeddings(
        openai_api_key=os.environ[f"EMBED_OPENAI_API_KEY"],
        deployment=os.environ[f"EMBED_OPENAI_DEPLOYMENT_NAME"],
        model=os.environ[f"EMBED_OPENAI_MODEL_NAME"],
        azure_endpoint=os.environ[f"EMBED_OPENAI_API_BASE"],
        openai_api_type=os.environ[f"EMBED_OPENAI_API_TYPE"],
        chunk_size=1000,
        max_retries=6,
        request_timeout=None,
        tiktoken_model_name=None,
    )
    else:
        return None
    
def load_llm():
    llm = None
    if os.environ[f"GENAI_TYPE"] == "OpenAI":
        llm = OpenAI(
            openai_api_key=os.environ[f"OPENAI_API_KEY"],
            model_name=os.environ[f"OPENAI_MODEL_NAME"],
            temperature=0.1,
            streaming=True
        )
    elif os.environ[f"GENAI_TYPE"] == "ChatOpenAI":
        llm = ChatOpenAI(
            openai_api_key=os.environ[f"OPENAI_API_KEY"],
            model_name=os.environ[f"OPENAI_MODEL_NAME"],
            temperature=0.1,
            streaming=True
        )
    elif os.environ[f"GENAI_TYPE"] == "AzureOpenAI":
        llm = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=os.environ[f"OPENAI_API_KEY"],
            model_name=os.environ["OPENAI_MODEL_NAME"],
            temperature=0.1,
            max_retries=12,
            streaming=True
        )
    elif os.environ[f"GENAI_TYPE"] == "AzureOpenAIChat":
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=os.environ[f"OPENAI_API_KEY"],
            model_name=os.environ["OPENAI_MODEL_NAME"],
            temperature=0.1,
            max_retries=12,
            model_kwargs={},
            streaming=True
        )
    return llm


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("./.env")

    llm = load_llm()
    print(llm.invoke("Where is Taj Mahal?"))
