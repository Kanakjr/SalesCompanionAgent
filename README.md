# SalesCompanionAgent

##### Description
SalesCompanionAgent is a comprehensive chat interface application tailored for sales assistants, empowering them to efficiently manage their sales activities. By integrating advanced AI models and a user-friendly interface, it facilitates the creation of sales plans, monitors goal achievements, organizes meetings, and assists in communication with clients. Designed to work with both structured and unstructured data, it is a versatile tool for any sales professional aiming to elevate their productivity and sales outcomes.

##### Features
- **Sales Plan Generation:** Automatically generates a sales plan based on input goals and historical data.
- **Progress Tracking:** Visualizes progress towards targets and goals in real-time.
- **Meeting Scheduler:** Assists in planning and scheduling meetings with target customers.
- **Email Drafting:** Provides templates and drafting assistance for communicating with clients.
- **Data Integration:** Seamlessly integrates with SQL databases for structured data and uses a vector store for unstructured data analysis.
- **User-Friendly Interface:** Built on the ChainLit framework for a responsive and intuitive user experience.

##### Technologies Stack
- Python: Primary programming language.
- ChromaDB, SQLite, FAISS: Used for data storage and retrieval.
- OpenAI and Azure: Provides language models and embedding capabilities.
- Dotenv: Manages environment variables.
- Langchain: A toolkit for building applications with language models.
- Chainlit: Used for building interactive web applications.

##### Getting Started
1. **Prerequisites**
   - A SQL database setup for structured data storage.
   - Access to a vector database for unstructured data.

2. **Installation**
   - Clone the repo: `git clone https://github.com/Kanakjr/SalesCompanionAgent.git`
   - Install python packages: `pip install -r requirnments.txt`

3. **Usage**
   - Start the application: `chainlit run app.py`
   - Navigate to `http://localhost:3000` to access the SalesCompanionAgent interface.
   - Follow the on-screen instructions to begin using the application for sales assistance.


### Main Components

#### 1. **app.py**
   - The frontend interface using Chainlit, which manages the session and user interaction in a chat-like interface.
   
#### 2. **sales_companion.py**
   - Acts as the central orchestrator for handling user queries. It decides which agent to use (SQL or RAG) based on the query context and the user's historical interactions.
   - Manages user sessions and retrieves user-specific data from a structured SQL database.

#### 3. **llm.py**
   - Manages loading and configuration of language models (LLMs) and embedding models from OpenAI or Azure, depending on the environment configuration.
   - Provides a caching mechanism for the language model to optimize performance.

#### 4. **structured.py**
   - Provides a wrapper for interacting with SQL databases, supports data import from Excel or CSV files, and manages schema introspection and query execution.

#### 5. **router_agent.py**
   - Determines the type of query (using SQL or Retrieval-Augmented Generation) and selects the appropriate agent for answering it based on predefined examples and the nature of the query.

#### 6. **sql_agent.py**
   - Executes SQL queries against a structured database and formats the results. This script uses language models to generate SQL queries based on user input.

#### 7. **rag_agent.py**
   - Handles retrieval of interaction notes and HCP names from the vectorstore, using similarity search based on the query and contextual information.

#### 8. **generate_agent.py**
   - Generates contextual outputs like meeting summaries, speaker notes, and follow-up emails based on the user query and available interaction notes.

#### 9. **create_vectorstore.py (Optional - you can connect to your on db)** 
   - Initializes a structured database and a vector-based database using ChromaDB for storing and retrieving interaction notes and healthcare professional (HCP) names.
   - Utilizes language models to generate interaction notes based on historical data, which are then stored in ChromaDB for future retrieval.