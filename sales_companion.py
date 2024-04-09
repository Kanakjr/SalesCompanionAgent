from structured import StructuredDatabase

structured_database_uri = "sqlite:///./data/SalesAssistant.db"

class SalesCompanion:
    def __init__(self, useremail):
        self.useremail = useremail
        self.db = StructuredDatabase.from_uri("sqlite:///./data/SalesAssistant.db")
        self.user_info = self.retrieve_user_info()

    def retrieve_user_info(self):
        df = self.db.df_from_sql_query(f'SELECT * FROM SalesRep where Email = "{self.useremail}"')
        if len(df)==1:
            return df.to_dict(orient='records')[0]
        return None
    
if __name__ == "__main__":
    salesCompanion = SalesCompanion(useremail='john.doe@example.com')
    print(salesCompanion.user_info)