import gradio as gr
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # type: ignore
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


# Load Huggingface tpken
HUGGINGFACEHUB_API_TOKEN = '#'
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN 
)


# Memory 
memory = ConversationBufferMemory()
conversation_buf = ConversationChain(
    llm=llm,
    memory = memory)

# Predict the response
def predict_response(message, history):
    answer = conversation_buf.predict(input=message)
    return answer

demo = gr.ChatInterface(predict_response, type="messages")

# Démarrer le serveur local 
demo.launch()
