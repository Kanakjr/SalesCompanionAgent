import chainlit as cl
from sales_companion import SalesCompanion


@cl.on_chat_start
def start():
    sales_companion = SalesCompanion(useremail="kanak.dahake@example.com")
    cl.user_session.set("sales_companion", sales_companion)


@cl.on_message
async def main(message: cl.Message):
    sales_companion = cl.user_session.get("sales_companion")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    res = await cl.make_async(sales_companion.run)(message.content, callbacks=[cb])
    await cl.Message(content=res).send()