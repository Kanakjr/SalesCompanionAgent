import os
import chainlit as cl
from sales_companion import SalesCompanion

def set_environment_variables(version):
    suffix = "_1" if version == "GPT-3.5" else "_2"
    os.environ["OPENAI_MODEL_NAME"] = os.environ[f"OPENAI_MODEL_NAME{suffix}"]
    if os.environ.get("GENAI_TYPE") == 'AzureOpenAIChat':
        os.environ["AZURE_ENDPOINT"] = os.environ[f"AZURE_ENDPOINT{suffix}"]
        os.environ["OPENAI_DEPLOYMENT_NAME"] = os.environ[f"OPENAI_DEPLOYMENT_NAME{suffix}"]
        os.environ["OPENAI_API_KEY"] = os.environ[f"OPENAI_API_KEY{suffix}"]

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
        ),
    ]

@cl.on_chat_start
def start():
    ## Set the llm model based on the chat profile    
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "GPT-3.5":
        set_environment_variables("GPT-3.5")
    elif chat_profile == "GPT-4":
        set_environment_variables("GPT-4")
    print(f"Chat Profile: {chat_profile}")
    print(f"MODEL_NAME: {os.environ.get('OPENAI_MODEL_NAME')}")

    ## Initialize the Sales Companion
    sales_companion = SalesCompanion(useremail="kanak.dahake@example.com")
    cl.user_session.set("sales_companion", sales_companion)


@cl.on_message
async def main(message: cl.Message):
    sales_companion = cl.user_session.get("sales_companion")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    response = await cl.make_async(sales_companion.run)(message.content, callbacks=[cb])

    answer = response['answer']
    agent = response['agent']

    if agent == "rag_agent":
        text_elements = []
        source_documents = response['source']
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx+1}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\n\n[Content Generated using following Sources: {', '.join(source_names)}]"
        else:
            answer += "\nNo sources found"
    
    # await cl.Message(content=res).send()
    await cl.Message(content=answer, elements=text_elements).send()