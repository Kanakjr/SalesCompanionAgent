from langchain_community.vectorstores import Chroma
from llm import load_llm, load_openai_embeddings

vector_db = Chroma(
    collection_name="interaction_notes",
    persist_directory="./data/vectorstore/",
    embedding_function=load_openai_embeddings(),
)

if __name__ == "__main__":
    query = "follow-up video call with Dr. Morgan Murphy"
    filter = {'$and': [{'SalesRep_ID': {'$eq': "T06"}},
                    {'HCP_ID': {'$eq': "HCP002"}}]}
    docs = vector_db.similarity_search(query,filter=filter)
    if docs:
        print(docs[0].page_content)
    else:
        print("No document found")
    