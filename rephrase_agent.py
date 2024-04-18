from langchain_core.pydantic_v1 import BaseModel, Field
from llm import load_llm
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from examples import examples
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig


def rephrase_question(query,conversation_history,callbacks=[]):
    # 
    prompt = PromptTemplate(
    template= """User query: {query}

Conversation History: 
{conversation_history}

Rephrase the user query based on conversation history as an standalone query.
Rephrased Query:""",
    input_variables=["query", "conversation_history"],
)
    # Chain the prompt template with the language model for processing
    chain = prompt | load_llm() | StrOutputParser()

    response = chain.invoke({"query": query,
                             "conversation_history":conversation_history,
                             },
                             config=RunnableConfig(run_name='rephrase_question',
                                                   callbacks=callbacks)
                            )
    return response

if __name__ == "__main__":
    # Conversation History - Examples
    query = "How many prescription has been written by Crystal Esparza?"
    conversation_history = ''
    print(rephrase_question(query,conversation_history))