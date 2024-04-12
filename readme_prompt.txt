## create_vectorstore.py
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

## llm.py
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


## router_agent.py
from langchain_core.pydantic_v1 import BaseModel, Field
from llm import load_llm
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from examples import examples
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable.config import RunnableConfig


# sql_agent_examples = [
#     {"input": example["input"], "description": example["description"]}
#     for example in examples
# ]

sql_agent_examples = "\n".join([ example["input"] for example in examples ])

rag_agent_examples = "\n".join(['Can you provide speaking notes for my Phone call/ Meeting with [HCP Name]?',
                      'Write me an email for follow-up with [HCP Name]',
                      'What are the key points to discuss with [HCP Name]?',
                      'Meeting notes for my last meeting with [HCP Name]'])

class RouterOutput(BaseModel):
    agent: str = Field(description="Agent to be used for answering the question")
    reason: str = Field(description="Reason for selecting the agent")

parser = JsonOutputParser(pydantic_object=RouterOutput)

def evaluate_question_type(query,callbacks=[]):
    # 
    prompt = PromptTemplate(
    template= """Given user query: {query}
Determine the type of question and select the appropriate agent to answer the question.
Available agents are:

1. 'sql_agent' used when we need an answer from a database that contains tables such as 'HCP' with fields like HCP_ID and HCP_Name, 'InteractionHistory', 'Prescription', 'PriorityLookup', and 'SalesRep'.
Example: {sql_agent_examples}

2. 'rag_agent' used for retrieving detailed narratives or context from unstructured data, such as meeting and interaction notes between a healthcare professional (HCP) and a sales representative.
Example: {rag_agent_examples}

{format_instructions}

Output:""",
    input_variables=["query", "sql_agent_examples", "rag_agent_examples"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
    # Chain the prompt template with the language model for processing
    chain = prompt | load_llm() | parser

    response = chain.invoke({"query": query,
                             "sql_agent_examples":sql_agent_examples,
                             "rag_agent_examples":rag_agent_examples
                             },
                             config=RunnableConfig(callbacks=callbacks)
                            )
    response = RouterOutput.parse_obj(response)
    return response

if __name__ == "__main__":
    # SQL Agent - Examples 
    query = "How many prescription has been written by Crystal Esparza?"
    print(evaluate_question_type(query))

    query = "How am I performing against his goal"
    print(evaluate_question_type(query))

    query = "Who should Kanak Dahake contact this week?"
    print(evaluate_question_type(query))

    # RAG Agent - Examples
    query = "Can you provide speaking notes for my Phone call/ Meeting with Dr. Morgan Murphy?"
    print(evaluate_question_type(query))

    query = "What are the key points to discuss with Dr. Morgan Murphy?"
    print(evaluate_question_type(query))

    query = "Write me an email for follow-up with Dr. Morgan Murphy"
    print(evaluate_question_type(query))

## sales_companion.py
from rag_agent import get_hcp_names,get_interaction_notes
from structured import StructuredDatabase
from sql_agent import SQLAgent
from router_agent import evaluate_question_type
from generate_agent import generate_content
from llm import load_llm
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import concurrent.futures
from langchain.schema.runnable.config import RunnableConfig


load_dotenv("./.env")
structured_database_uri = "sqlite:///./data/SalesAssistant.db"

class SalesCompanion:
    def __init__(self, useremail):
        self.useremail = useremail
        self.db = StructuredDatabase.from_uri(structured_database_uri)
        self.user_info = self.retrieve_user_info()
        self.llm = load_llm()
        self.sql_agent = SQLAgent(llm=self.llm, db=self.db, user_info=self.user_info)

    def retrieve_user_info(self):
        df = self.db.df_from_sql_query(
            f'SELECT * FROM SalesRep where Email = "{self.useremail}"'
        )
        if len(df) == 1:
            return df.to_dict(orient="records")[0]
        return None
    
    def get_hcp_details(self, query, callbacks=[]):
        hcp_names = get_hcp_names(query)
        if hcp_names != "No document found":
            prompt = ChatPromptTemplate.from_template(
                """Given user query: {query}
                and a list of HCP names: {hcp_names}
                Figure out if the user query is related to any of the HCP names provided. If so, return only the HCP Name from provided list of HCP names as output.
                If not, return 'No match found' as output.'
                output:"""
            )
            # Chain the prompt template with the language model for processing
            chain = prompt | self.llm
            response = chain.invoke({"query": query, "hcp_names": hcp_names},
                                    config=RunnableConfig(callbacks=callbacks)
                                    ).content
            if response != "No match found":
                hcp_details = self.db.df_from_sql_query(f'SELECT * FROM HCP where HCP_Name = "{response}"').to_dict(orient="records")
                if len(hcp_details) == 1:
                    return hcp_details[0]
        return None
    
    def invoke(self, input):
        return self.sql_agent.invoke(input=input)

    def run(self, input, callbacks=[]):
        evaluate_question_type_response = evaluate_question_type(input, callbacks=callbacks)
        hcp_details = self.get_hcp_details(query=input, callbacks=callbacks)
        
        if evaluate_question_type_response.agent == "rag_agent":
            if hcp_details:
                print(f"Fetching interaction notes for HCP: {hcp_details['HCP_Name']} against SalesRep: {self.user_info['SalesRep_ID']}")
                interaction_notes = get_interaction_notes(query=input, SalesRep_ID=self.user_info["SalesRep_ID"], HCP_ID=hcp_details["HCP_ID"])
            else:
                interaction_notes = get_interaction_notes(query=input, SalesRep_ID=self.user_info["SalesRep_ID"])
            
            generate_output = generate_content(query=input,
                                        salesrep_details=self.user_info,
                                        hcp_details=hcp_details,
                                        interaction_notes=interaction_notes, 
                                        callbacks=callbacks)
            return f"{generate_output.response}"
                
                
        
        elif evaluate_question_type_response.agent == "sql_agent":
            return self.sql_agent.run(input=input, callbacks=callbacks)


    # Example of running multiple functions concurrently ###############
    # def run(self, input, callbacks=[]):
    #     # Function to be executed in parallel
    #     def fetch_hcp_details():
    #         hcp_details = self.get_hcp_details(query=input, callbacks=callbacks)
    #         if hcp_details:
    #             print(f"Fetching interaction notes for HCP: {hcp_details['HCP_Name']} against SalesRep: {self.user_info['SalesRep_ID']}")
    #             interaction_notes = get_interaction_notes(query=input, SalesRep_ID=self.user_info["SalesRep_ID"], HCP_ID=hcp_details["HCP_ID"])
    #             return hcp_details,interaction_notes
    #         else:
    #             return "No HCP details found","No interaction notes found"

    #     def fetch_sql_agent_run():
    #         return self.sql_agent.run(input=input, callbacks=callbacks)
    #         # return ''

    #     # Create a ThreadPoolExecutor to manage concurrent execution
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         # Schedule both functions to run concurrently
    #         hcp_future = executor.submit(fetch_hcp_details)
    #         sql_future = executor.submit(fetch_sql_agent_run)

    #         # Wait for both futures to complete
    #         hcp_details,interaction_notes = hcp_future.result()
    #         agent_answer = sql_future.result()

    #     # Combine the results and return
    #     return f"{agent_answer}\n\nHCP Details: {hcp_details}\n\nInteraction Notes: {interaction_notes}"
    


if __name__ == "__main__":
    sales_companion = SalesCompanion(useremail="kanak.dahake@example.com")
    
    #print(f"User Info: {sales_companion.user_info}")

    # hcp_details = sales_companion.get_hcp_details("follow-up video call with Dr. Luffy")
    # print(f"HCP Details: {hcp_details}")

    # hcp_details = sales_companion.get_hcp_details("follow-up video call with Dr. Morgan Murphy")
    # print(f"HCP Details: {hcp_details}")

    # response = sales_companion.invoke("How am I performing against my goals?")
    # print(response)

    response = sales_companion.run("What are the key points to discuss with Dr. Morgan Murphy?")
    print(response)

    # response = sales_companion.run("How many prescription has been written by Dr. Morgan Murphy?")
    # print(response)
    


## rag_agent.py
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

## examples.py
examples = [
    {
        "input": "How am I performing against his goal",
        "description": "When Sales Rep want to know about his performance",
        "query": "WITH TargetCalls AS (\n    SELECT \n\t\tsr.Name,\n        sr.SalesRep_ID, \n        h.Priority,\n        SUM(pl.Calls) AS Target_Calls\n    FROM \n        main.SalesRep sr\n    JOIN \n        main.HCP h ON sr.SalesRep_ID = h.Territory_ID\n    JOIN \n        main.PriorityLookup pl ON h.Priority = pl.Priority\n\tWHERE\n\t\tsr.Name = 'Ethan Miller'\n    GROUP BY \n        sr.Name,sr.SalesRep_ID, sr.Territory_ID, h.Priority\n), TotalCallsMade AS (\n    SELECT \n        ih.SalesRep_ID, \n\t\th.Priority,\n        COUNT(ih.History_ID) AS Total_Calls_Made\n    FROM \n        main.InteractionHistory ih\n\t\tJOIN \n        main.HCP h ON h.HCP_ID = ih.HCP_ID\n\t\tJOIN\n\t\tmain.SalesRep sr ON sr.SalesRep_ID = ih.SalesRep_ID\n\tWHERE\n\t\tsr.Name = 'Ethan Miller'\n    GROUP BY \n        ih.SalesRep_ID, h.Priority\n)\nSELECT \n    tc.Name,\n\ttc.SalesRep_ID,\n    tc.Priority,\n\ttc.Target_Calls,\n    COALESCE(tcm.Total_Calls_Made, 0) AS Total_Calls_Made,\n    COALESCE((COALESCE(tcm.Total_Calls_Made, 0) * 100.0) / NULLIF(tc.Target_Calls, 0), 0) AS Percent_Achieved\nFROM \n    TargetCalls tc\nLEFT JOIN \n    TotalCallsMade tcm ON tc.SalesRep_ID = tcm.SalesRep_ID and tc.Priority = tcm.Priority\n\torder by tc.SalesRep_ID,tc.Priority;",
        "format_instruction": "**_{{Name}} Performance Report:_**  \n- **{{Priority}}:** Made **{{Total_Calls_Made}}** calls out of **{{Target_Calls}}** calls target and achieved **{{Percent_Achieved}}**.  \n"
    },
    {
        "input": "What I should know about Doctor [HCP Name]",
        "description": "When Sales Rep wants to know about doctors",
        "query": "Select HCP_Name,Speciality,Phone_No,Email,Account_Type,Account_Name from [main].[HCP] Where HCP_Name = 'William Davis';",
        "format_instruction": "{{HCP_Name}} is a {{Speciality}} specialist at {{Account_Name}}. You can contact him via phone at {{Phone_No}} or email at {{Email}}.\n  - **HCP Name**: {{HCP_Name}}\n  - **Specialty**: {{Speciality}}\n  - **Phone Number**: {{Phone_No}}\n  - **Email**: {{Email}}\n  - **Account Type**: {{Account_Type}}\n  - **Account Name**: {{Account_Name}}"
    },
    {
        "input": "Who should [SalesRep] contact this week?",
        "description": "Sales Rep is asking for meeting plan to whom he should meet",
        "query": "WITH LastInteraction AS (\n    SELECT ih.HCP_ID, count(*) Number_of_Interaction,MAX(ih.Contact_Date) AS Last_Interaction_Date\n    FROM \n        main.InteractionHistory as ih\n    GROUP BY ih.HCP_ID\n)\nSELECT \n    sr.Name,\n    h.HCP_ID,\n\th.HCP_Name,\n    h.Priority,\n    pl.Calls Target,\n\tCOALESCE(li.Number_of_Interaction,0) Number_of_Interaction,\n\tCOALESCE(li.Number_of_Interaction,0)*100 / pl.Calls AS '%Achieved',\n    DATEDIFF(DAY, COALESCE(li.Last_Interaction_Date,DATEADD(QUARTER,DATEDIFF(QUARTER,0,GETDATE()),0)), GETDATE()) AS Days_Since_Last_Interaction,\n    pl.Calls * DATEDIFF(DAY, COALESCE(li.Last_Interaction_Date,DATEADD(QUARTER,DATEDIFF(QUARTER,0,GETDATE()),0)) ,GETDATE()) AS Combined_Score\nFROM \n\tmain.SalesRep sr \n\tLEFT JOIN main.HCP h  ON sr.SalesRep_ID = h.Territory_ID\n\tLEFT JOIN LastInteraction li  ON li.HCP_ID = h.HCP_ID\n\tJOIN main.PriorityLookup pl on h.Priority = pl.Priority\nWHERE \n\tsr.Name = 'Ethan Miller' \nORDER BY \n    Combined_Score DESC;",
        "format_instruction": "Here is plan for [SalesRep]:  \n        1. Schedule a meeting with **{{HCP_Name}}** a priority **{{Priority}}** contact. It has been **{{Days_Since_Last_Interaction}}** since your last interaction. Your goal is to meet **{{Target}}** this quarter, and you have currently completed **{{Number_of_Interaction}}** of these meetings.  \n        2. Schedule a meeting with **{{HCP_Name}}** a priority **{{Priority}}** contact. It has been **{{Days_Since_Last_Interaction}}** since your last interaction. Your goal is to meet **{{Target}}** this quarter, and you have currently completed **{{Number_of_Interaction}}** of these meetings.  \n        "
    },
    {
        "input": "How many prescription has been written by [HCP Name]?",
        "description": "When Sales Rep want to know the prescription from given doctor",
        "query": "Select hcp.HCP_Name, SUM(P.TRx) Total_Number_Of_Prescription, SUM(P.NRx) Number_Of_New_Prescription from [main].[Prescription] P join main.HCP hcp on P.[HCP ID] = hcp.HCP_ID join [main].[SalesRep] sr on sr.SalesRep_ID = P.Territory_ID where hcp.HCP_Name = 'Crystal Esparza' Group by hcp.HCP_Name;",
        "format_instruction": ""
    },
    {
        "input": "Which doctors are assigned to me and what are their priority?",
        "description": "When Sales Rep want to know the doctors assigned to them",
        "query": "Select hcp.HCP_Name, Priority\nfrom main.HCP hcp join [main].[SalesRep] sr on hcp.Territory_ID = sr.Territory_ID\nwhere sr.Name = 'Kanak Dahake';",
        "format_instruction": ""
    }
]


## structured.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from copy import deepcopy
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    MetaData,
    Table,
    inspect,
    select,
    func,
    Numeric,
    FLOAT,
    INTEGER,
)
import re
from langchain_community.utilities.sql_database import SQLDatabase
from typing import Any, List, Tuple, Dict

from random import sample

import warnings

warnings.filterwarnings("ignore")


def drop_byte_columns(df):
    # get list of columns with byte type
    byte_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object"
        and df[col].size > 0
        and isinstance(df[col][0], bytes)
    ]
    # drop byte columns
    if byte_cols:
        df = df.drop(columns=byte_cols)
    return df


class StructuredDatabase(SQLDatabase):

    schema_str: str = None

    @classmethod
    def from_excel_or_csv(cls, filepath, sqlite_in_memory=True):
        basename = os.path.basename(filepath)
        basename_wo_extention, file_extention = os.path.splitext(basename)
        filepath_wo_extention = os.path.splitext(filepath)[0]

        if not os.path.exists(filepath):
            raise Exception(f"{filepath} do not exist")

        # load data from file
        if file_extention == ".xlsx":
            try:
                tables = pd.read_excel(filepath, sheet_name=None, header=None)
            except:
                raise Exception(f"{filepath} is corrupt")

            for tablename, table in list(tables.items()):
                sheet_tables = cls._get_tables_using_skimage(table)
                sheet_tables = sorted(
                    sheet_tables,
                    key=lambda table: table.shape[0] * table.shape[1],
                    reverse=True,
                )
                if len(sheet_tables) >= 1:
                    table = sheet_tables[0]
                    table = cls._add_header_to_table(table, 1)
                    tables[tablename] = table

        elif file_extention == ".csv":
            try:
                table = pd.read_csv(filepath)
            except:
                Exception(f"{filepath} is corrupt")
            tables = {}
            if len(table) >= 1:
                tables[basename_wo_extention] = table

        # clean data suitable for sql and gen ai and correct the data types
        tables_cleaned = {}
        for tablename, table in tables.items():
            table = table.loc[:, ~table.columns.str.lower().duplicated()]
            table.columns = table.columns.str.lower()
            table.columns = table.columns.str.strip()
            table.columns = table.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

            for column in table.columns:
                try:
                    table[column] = pd.to_numeric(table[column])
                except:
                    pass

            tablename_new = re.sub(r"[^a-zA-Z0-9_]", "_", tablename)
            tables_cleaned[tablename_new] = table

        # Populate SQLite with data from dictionary and store metadata
        if sqlite_in_memory:
            database_url = "sqlite:///:memory:"
        else:
            database_url = f'sqlite:///{filepath_wo_extention + ".db"}'
            if os.path.exists(filepath_wo_extention + ".db"):
                os.remove(filepath_wo_extention + ".db")

        engine = create_engine(database_url)
        metadata = MetaData()
        for tablename, table in tables_cleaned.items():
            # Create table
            cols = []

            for column in table.columns:
                s = table[column]
                s_dtype = s.dtype.kind
                if s_dtype == "i":
                    sqlalchemy_dtype_object = Integer
                elif s_dtype == "f":
                    sqlalchemy_dtype_object = Float
                elif s_dtype == "b":
                    sqlalchemy_dtype_object = Boolean
                elif s_dtype == "O":
                    sqlalchemy_dtype_object = String
                else:
                    raise Exception()
                is_unique = s.is_unique
                cols.append(Column(column, sqlalchemy_dtype_object, unique=is_unique))

            sqlalchemy_table = Table(tablename, metadata, *cols)
            metadata.create_all(engine)

            # Insert data
            # table.to_sql(tablename, engine, if_exists='replace', index=False)
            with engine.connect() as conn:
                conn.execute(sqlalchemy_table.insert(), table.to_dict(orient="records"))

        return cls(engine)

    def df_from_sql_query(self, query):
        return pd.read_sql(query, self._engine)

    def fetch_distinct_row_values(
        self, column_list: List[Tuple[str, str]], num_max_distinct=12
    ) -> Dict[Tuple[str, str], List[Any]]:
        column_list = eval(column_list)
        column_values_dict = {}
        with self._engine.connect() as connection:
            for table_name, column_name in column_list:
                # Ensure the table and column exist
                if table_name not in self._metadata.tables:
                    raise ValueError(f"Table {table_name} not found in the database")
                table = self._metadata.tables[table_name]
                if column_name not in table.columns:
                    raise ValueError(
                        f"Column {column_name} not found in table {table_name}"
                    )

                query = select(table.columns[column_name]).distinct()
                distinct_values = connection.execute(query).fetchall()
                distinct_values = [
                    item[0] for item in distinct_values
                ]  # Flatten the list
                # If numeric column and distinct values greater than num_max_distinct, fetch min and max
                if (
                    isinstance(table.columns[column_name].type, Numeric)
                    or isinstance(table.columns[column_name].type, INTEGER)
                    or isinstance(table.columns[column_name].type, FLOAT)
                ) and len(distinct_values) > num_max_distinct:
                    min_value = connection.execute(
                        select(func.min(table.columns[column_name]))
                    ).scalar()
                    max_value = connection.execute(
                        select(func.max(table.columns[column_name]))
                    ).scalar()
                    # column_values_dict[(table_name, column_name)] = [min_value, max_value]
                    column_values_dict[(table_name, column_name)] = (
                        f"Numerical values from min {min_value} to max {max_value}"
                    )
                # If non-numeric and distinct values greater than num_max_distinct, fetch random num_max_distinct values
                elif len(distinct_values) > num_max_distinct:
                    column_values_dict[(table_name, column_name)] = sample(
                        distinct_values, num_max_distinct
                    )
                else:
                    column_values_dict[(table_name, column_name)] = distinct_values

        return str(column_values_dict)

    def get_schema_str(self):
        if self.schema_str is not None:
            return self.schema_str
        schema_str = ""
        inspector = inspect(self._engine)
        table_names = inspector.get_table_names()

        for table_name in table_names:
            schema_str += f"TableName: {str(table_name)}\n"
            schema_str += "FieldName | FieldType | IsUnique\n"
            columns = inspector.get_columns(table_name)

            unique_keys = [
                uk["column_names"]
                for uk in inspector.get_unique_constraints(table_name)
            ]
            unique_columns = [col for sublist in unique_keys for col in sublist]

            for column in columns:
                # Getting data type of column
                FieldName = column["name"]
                FieldType = str(column["type"])
                IsUnique = column["name"] in unique_columns

                schema_str += f"{FieldName} | {FieldType} | {IsUnique}\n"
            schema_str += "\n"
        self.schema_str = schema_str
        return schema_str

    @staticmethod
    def _get_tables_using_skimage(df, using_empty_row_col=False):
        if using_empty_row_col:
            binary_rep = np.ones_like(df)
            for i in range(len(df.index)):
                if df.iloc[i, :].isnull().all():
                    binary_rep[i, :] = 0
            for j in range(len(df.columns)):
                if df.iloc[:, j].isnull().all():
                    binary_rep[:, j] = 0
        else:
            binary_rep = np.array(df.notnull().astype("int"))

        l = label(binary_rep)
        tables = []
        for s in regionprops(l):
            table = df.iloc[s.bbox[0] : s.bbox[2], s.bbox[1] : s.bbox[3]]
            tables.append(table)

        return tables

    @staticmethod
    def _add_header_to_table(table, num_headers):
        result_table = deepcopy(table)
        column_headers = [
            tuple(table.iloc[i, j] for i in range(num_headers))
            for j in range(table.shape[1])
        ]

        if num_headers == 1:
            result_table.columns = list(table.iloc[0, :])
        elif num_headers >= 2:
            result_table.columns = pd.MultiIndex.from_tuples(column_headers)

        result_table = result_table.iloc[num_headers:, :]
        return result_table


# %%

if __name__ == "__main__":

    filepath = "./datasets/AdventureWorks.xlsx"

    # Load data from excel or csv file, sqlite_in_memory=True will store data in memory
    # db = StructuredDatabase.from_excel_or_csv(filepath, sqlite_in_memory=False)

    # Load data from sqlite database
    db = StructuredDatabase.from_uri("sqlite:///./data/SalesAssistant.db")

    # Get schema string
    schema_str = db.get_schema_str()
    print(schema_str)

    # Get data from database
    result = db.run("SELECT * FROM SalesRep")
    print(result)
    
    # Get data from database as pandas dataframe
    df = db.df_from_sql_query("SELECT * FROM SalesRep")
    print(df)
    
    # Fetch distinct row values
    distinct_row_values = db.fetch_distinct_row_values("[('SalesRep', 'Name')]")
    print(distinct_row_values)

## app.py
import chainlit as cl
from sales_companion import SalesCompanion


@cl.on_chat_start
def start():
    sales_companion = SalesCompanion(useremail="kanak.dahake@example.com")
    cl.user_session.set("sales_companion", sales_companion)


@cl.on_message
async def main(message: cl.Message):
    sales_companion = cl.user_session.get("sales_companion")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    res = await cl.make_async(sales_companion.run)(message.content, callbacks=[cb])
    await cl.Message(content=res).send()

## generate_agent.py
from langchain_core.pydantic_v1 import BaseModel, Field
from llm import load_llm
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from examples import examples
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable.config import RunnableConfig

## Ganerate agent will generate meetting summary, speaker notes, follow-up email, key points to discuss with HCP etc
## We will provide the agent with the user query
## We will also provide the interaction notes and HCP names to the agent to help in context understanding
## The agent will determine the task like speaker notes, follow-up email, key points to discuss with HCP and generate the output
## The agent will also provide the reason for selecting the task based on the user query

class GenerateOutput(BaseModel):
    response: str = Field(description="Content generated like meeting summary, speaker notes, follow-up email, key points to discuss with HCP. It shoud be markdown formatted.")
    task: str = Field(description="Task to be performed. Example: speaker notes, follow-up email, key points to discuss, meeting summary, etc")
    reason: str = Field(description="Reason for selecting the task")

parser = JsonOutputParser(pydantic_object=GenerateOutput)

def generate_content(query,salesrep_details,hcp_details,interaction_notes, callbacks=[]):
    # 
    prompt = PromptTemplate(
    template= """Given user query: {query}
Determine the type of question and select the appropriate task to be performed.

These are the interactions notes and HCP names available to help in context understanding:

Sales Representative Details: {salesrep_details}
HCP Details: {hcp_details}

Interaction Notes: {interaction_notes}

{format_instructions}

Output:""",
    input_variables=["query", "sql_agent_examples", "rag_agent_examples"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
    # Chain the prompt template with the language model for processing
    chain = prompt | load_llm() | parser

    response = chain.invoke({"query": query,
                             "salesrep_details":salesrep_details,
                             "hcp_details":hcp_details,
                             "interaction_notes":interaction_notes,
                             },
                             config=RunnableConfig(callbacks=callbacks)
                            )
    response = GenerateOutput.parse_obj(response)
    return response

if __name__ == "__main__":
    from rag_agent import get_interaction_notes

    salesrep_details = {'SalesRep_ID': 'T09', 'Name': 'Kanak Dahake', 'Team': 'Regional Team A', 'Role': 'Sales Rep', 'Territory_ID': 'T09', 'Email': 'kanak.dahake@example.com'}
    hcp_details = {'HCP_ID': 'HCP035', 'HCP_Name': 'Morgan Murphy', 'Account_Type': 'Clinic', 'Account_Name': 'Douglas, Drake and Olsen Clinic', 'Email': 'morgan.murphy@healthmail.com', 'Phone_No': '493-871-2581x1148', 'Speciality': 'Cardiology', 'Priority': 'C', 'Territory_ID': 'T09'}
    
    query = "What are the key points to discuss with Dr. Morgan Murphy?"
    interaction_notes = get_interaction_notes(query=query, SalesRep_ID=salesrep_details["SalesRep_ID"], HCP_ID=hcp_details["HCP_ID"])
    generated_content = generate_content(query, salesrep_details, hcp_details, interaction_notes)
    print(f"Generated Content: {generated_content}")

    query = "Write me an email for follow-up with Dr. Morgan Murphy"
    interaction_notes = get_interaction_notes(query=query, SalesRep_ID=salesrep_details["SalesRep_ID"], HCP_ID=hcp_details["HCP_ID"])
    generated_content = generate_content(query, salesrep_details, hcp_details, interaction_notes)
    print(f"Generated Content: {generated_content}")

## readme_prompt.py
import os

# Set the directory where your .py files are stored
directory = './'

# Name of the output file
output_file = 'readme_prompt.txt'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter and process only .py files
        for filename in files:
            if filename.endswith('.py'):
                # Write the filename as a header in the output file
                outfile.write(f"## {filename}\n")
                
                # Open and read the content of the .py file
                with open(os.path.join(root, filename), 'r') as file:
                    content = file.read()
                    # Write the content of the .py file to the output file
                    outfile.write(content + '\n\n')

    outfile.write('Given the above code snippets, your task is to create a detailed README file that explains the functionality of each code snippet, how they interact with each other, and how they contribute to the overall functionality of the system. You can use the code snippets as a reference to create the README file.\n')


## sql_agent.py
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from examples import examples
from llm import load_openai_embeddings

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

User Information: {user_info}

Here are some examples of user inputs and their corresponding SQL queries:"""


class SQLAgent:
    def __init__(self, llm, db, user_info):
        self.db = db
        self.llm = (llm,)
        self.user_info = user_info

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            load_openai_embeddings(),
            FAISS,
            k=5,
            input_keys=["input"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nDescription:{description}\nSQL query: {query}\nFormat Instruction: {format_instruction}"
            ),
            input_variables=["input", "dialect", "top_k", "user_info"],
            prefix=system_prefix,
            suffix="",
        )

        sql_agent_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        self.agent = create_sql_agent(
            llm=llm,
            db=db,
            prompt=sql_agent_prompt,
            verbose=True,
            agent_type="openai-tools",
        )

    def invoke(self, input):
        return self.agent.invoke({"input": input, "user_info": self.user_info})

    def run(self, input, callbacks=[]):
        return self.agent.run(
            {"input": input, "user_info": self.user_info}, callbacks=callbacks
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from structured import StructuredDatabase
    from llm import load_llm

    load_dotenv("./.env")
    llm = load_llm()

    db = StructuredDatabase.from_uri("sqlite:///./data/SalesAssistant.db")
    print(db.get_schema_str())

    user_info = {
        "SalesRep_ID": "T01",
        "Name": "John Doe",
        "Team": "Regional Team C",
        "Role": "Sales Rep",
        "Territory_ID": "T01",
        "Email": "john.doe@example.com",
    }
    
    sql_agent = SQLAgent(llm=llm, db=db, user_info=user_info)

    # sql_agent.invoke("How am I performing against my goals?")
    # sql_agent.invoke("Which doctors are assigned to me and what are their priority?")
    # sql_agent.invoke("How many prescription has been written by Dr. Robert Wilson?")
    # sql_agent.invoke("Who should I contact this week")
    # sql_agent.invoke("Write an email to Dr. Robert Wilson?")

    # sql_agent.invoke("Can you Draft an email to Tanya Lewis?")



Given the above code snippets, your task is to create a detailed README file that explains the functionality of each code snippet, how they interact with each other, and how they contribute to the overall functionality of the system. You can use the code snippets as a reference to create the README file.
