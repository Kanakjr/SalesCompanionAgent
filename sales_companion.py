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
            if hcp_details:
                input = f"{input} \n\n HCP Details: {hcp_details['HCP_Name']}"
            response = self.sql_agent.run(input=input, callbacks=callbacks)
            return response


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
    
