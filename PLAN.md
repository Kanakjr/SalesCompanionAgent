

### Workflow

1. Sales Assistant logs in to the app using his email
2. Do SQL query to get the Sales Assistant details using his email: Select * from salesassistant where salesassistant.email = {email}
3. User asks a question > Rephrase the question LLM
4. Rephrased Question > Router LLM
Decides the intent or if the question is relevant with the use case:
Possible outcomes:
Junk
Irrelevant question > suggest recommened question
relevant question > detect the intent and other details
5. If customer details is required use SQL agent get details
6. If RAG is required run the RAG_agent
7. If generation is required use the 
8. Print inter steps and results to UI