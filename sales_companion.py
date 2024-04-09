from structured import StructuredDatabase
from sql_agent import SQLAgent
from llm import load_llm

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

    def invoke(self, input):
        return self.sql_agent.invoke(input=input)

    def run(self, input, callbacks=[]):
        return self.sql_agent.run(input=input, callbacks=callbacks)


if __name__ == "__main__":
    sales_companion = SalesCompanion(useremail="john.doe@example.com")
    print(sales_companion.user_info)

    response = sales_companion.invoke("How am I performing against my goals?")
    print(response)
