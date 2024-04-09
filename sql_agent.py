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
                "User input: {input}\nSQL query: {query}\nFormat Instruction: {format_instruction}"
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

    load_dotenv("./.env")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    db = StructuredDatabase.from_uri("sqlite:///./data/SalesAssistant.db")

    user_info = {
        "SalesRep_ID": "T01",
        "Name": "John Doe",
        "Team": "Regional Team C",
        "Role": "Sales Rep",
        "Territory_ID": "T01",
        "Email": "john.doe@example.com",
    }
    sql_agent = SQLAgent(llm=llm, db=db, user_info=user_info)

    sql_agent.invoke("How am I performing against my goals?")
    # sql_agent.invoke("Which doctors are assigned to me and what are their priority?")
    # sql_agent.invoke("How many prescription has been written by Dr. Robert Wilson?")
    # sql_agent.invoke("Who should I contact this week")
    # sql_agent.invoke("Write an email to Dr. Robert Wilson?")
