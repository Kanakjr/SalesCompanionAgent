from rag_agent import get_hcp_names
from structured import StructuredDatabase
from sql_agent import SQLAgent
from llm import load_llm
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

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
    
    def get_hcp_details(self, query):
        hcp_names = get_hcp_names(query)
        if hcp_names != "No document found":
            prompt = ChatPromptTemplate.from_template(
                """Given user query: {query}
                and a list of HCP names: {hcp_names}
                Figure out if the user query is related to any of the HCP names provided. If so, return only the HCP Name as output.
                If not, return 'No match found' as output.'
                output:"""
            )
            # Chain the prompt template with the language model for processing
            chain = prompt | self.llm
            response = chain.invoke({"query": query, "hcp_names": hcp_names}).content
            if response != "No match found":
                hcp_details = self.db.df_from_sql_query(f'SELECT * FROM HCP where HCP_Name = "{response}"').to_dict(orient="records")
                if len(hcp_details) == 1:
                    return hcp_details[0]
        return None
    
    def invoke(self, input):
        return self.sql_agent.invoke(input=input)

    def run(self, input, callbacks=[]):
        return self.sql_agent.run(input=input, callbacks=callbacks)


if __name__ == "__main__":
    sales_companion = SalesCompanion(useremail="john.doe@example.com")
    print(f"User Info: {sales_companion.user_info}")

    hcp_details = sales_companion.get_hcp_details("follow-up video call with Dr. Luffy")
    print(f"HCP Details: {hcp_details}")

    hcp_details = sales_companion.get_hcp_details("follow-up video call with Dr. Morgan Murphy")
    print(f"HCP Details: {hcp_details}")

    # response = sales_companion.invoke("How am I performing against my goals?")
    # print(response)

    # response = sales_companion.invoke("Can you Draft an email to Tanya Lewis?")
    # print(response)
