from langchain_community.vectorstores import Chroma
from llm import load_llm, load_openai_embeddings

vectordb_interaction_notes = Chroma(
    collection_name="interaction_notes",
    persist_directory="./data/vectorstore/",
    embedding_function=load_openai_embeddings(),
)

vectordb_hcp_names = Chroma(
    collection_name="hcp_names",
    persist_directory="./data/vectorstore/",
    embedding_function=load_openai_embeddings(),
)



def get_interaction_notes(query, SalesRep_ID=None, HCP_ID=None):
    # Create filter based on SalesRep_ID and HCP_ID. If both are None, filter is empty
    # if both are given use and operator. If either is given use or operator
    if SalesRep_ID and HCP_ID:
        filter = {"$and": [{"SalesRep_ID": {"$eq": SalesRep_ID}}, {"HCP_ID": {"$eq": HCP_ID}}]}
    elif SalesRep_ID:
        filter = {"SalesRep_ID": {"$eq": SalesRep_ID}}
    elif HCP_ID:
        filter = {"HCP_ID": {"$eq": HCP_ID}}
    else:
        filter = {}
    docs = vectordb_interaction_notes.similarity_search(query, filter=filter)
    if docs:
        return docs
    else:
        return "No document found"
    
def get_hcp_names(query,filter={}):
    docs = vectordb_hcp_names.similarity_search(query, filter=filter)
    if docs:
        return docs
    else:
        return "No document found"

if __name__ == "__main__":
    query = "follow-up video call with Dr. Morgan Murphy"
    
    # interaction_notes = get_interaction_notes(query, SalesRep_ID="T09", HCP_ID="HCP035")
    # print(f"Interaction Notes: {interaction_notes}")

    # hcp_name = get_hcp_names(query)
    # print(f"HCP Name: {hcp_name}")