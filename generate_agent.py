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
                             config=RunnableConfig(run_name='generate_content',
                                                   callbacks=callbacks)
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