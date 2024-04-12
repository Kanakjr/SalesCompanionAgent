import chromadb
from structured import StructuredDatabase
from langchain_core.prompts import ChatPromptTemplate
from llm import load_llm, load_openai_embeddings
from dotenv import load_dotenv

# Load environment variables, which may include API keys and database paths
load_dotenv("./.env")

# Initialize database connection to store and retrieve sales interactions
structured_database_uri = "sqlite:///./data/SalesAssistant.db"
db = StructuredDatabase.from_uri(structured_database_uri)

# Initialize ChromaDB client for vector-based storage and retrieval of interaction notes
chroma_client = chromadb.PersistentClient(path="./data/vectorstore/")

def create_interaction_notes(llm, openai_embedding, collection_name):
    # Template for generating interaction notes based on details of a sales rep's interaction with a doctor (HCP)
    prompt = ChatPromptTemplate.from_template(
        """Given the details of an interaction between a sales representative and a doctor (HCP).
        Interaction Details: {interaction_detail}                                  
        You are the sales rep from the interaction and your task is to generate notes for the given interaction in about 100-200 words.
        Interaction Notes:"""
    )
    # Chain the prompt template with the language model for processing
    chain = prompt | llm

    # Query the database for historical interaction details
    interaction_history = db.df_from_sql_query(
        """SELECT IH.*, 
                HCP.HCP_Name, 
                HCP.Account_Type AS HCP_Account_Type, 
                HCP.Account_Name AS HCP_Account_Name, 
                HCP.Email AS HCP_Email, 
                HCP.Phone_No AS HCP_Phone_No, 
                HCP.Speciality AS HCP_Speciality, 
                SR.Name AS SalesRep_Name, 
                SR.Team AS SalesRep_Team,
                SR.Role AS SalesRep_Role,
                SR.Email AS SalesRep_Email
            FROM InteractionHistory IH
            LEFT JOIN HCP ON IH.HCP_ID = HCP.HCP_ID
            LEFT JOIN SalesRep SR ON IH.SalesRep_ID = SR.SalesRep_ID;
        """
    ).to_dict(orient="records")

    # Generate interaction notes for each historical interaction detail
    responses = chain.batch([{"interaction_detail": interaction_detail} for interaction_detail in interaction_history])

    # Create or get a collection in ChromaDB to store the generated interaction notes
    collection = chroma_client.create_collection(name=collection_name)
    # Add the generated notes to the collection, including embeddings for future retrieval, documents, and metadata
    collection.add(
        embeddings=[openai_embedding.embed_query(response.content) for response in responses],
        documents=[response.content for response in responses],
        metadatas=[interaction_detail for interaction_detail in interaction_history],
        ids=[interaction_detail["History_ID"] for interaction_detail in interaction_history],
    )

def create_hcp_names(llm, openai_embedding, collection_name):
    hcp_list = db.df_from_sql_query(
        """SELECT *
        FROM HCP;
        """
    ).to_dict(orient="records")

    collection = chroma_client.create_collection(name=collection_name)
    # Add the generated notes to the collection, including embeddings for future retrieval, documents, and metadata
    collection.add(
        embeddings=[openai_embedding.embed_query(hcp['HCP_Name']) for hcp in hcp_list],
        documents=[hcp['HCP_Name'] for hcp in hcp_list],
        metadatas=[hcp for hcp in hcp_list],
        ids=[hcp['HCP_ID'] for hcp in hcp_list],
    )


if __name__ == "__main__":
    # Load language model and embeddings necessary for generating and storing interaction notes
    llm = load_llm()
    openai_embedding = load_openai_embeddings()

    # Execute the function to create interaction notes in the specified collection
    collection_name = "interaction_notes"
    create_interaction_notes(llm, openai_embedding, collection_name)

    # Execute the function to create HCP names in the specified collection
    collection_name = "hcp_names"
    create_hcp_names(llm, openai_embedding, collection_name)