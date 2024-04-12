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