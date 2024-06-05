from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import transformers
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
import os
import streamlit as st
from streamlit_chat import message

huggingfacehub_api_token = enter your token here

def initialize_session_state():
    if 'history' not in st.session_state:
         st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me any question from the Bhagavad Geetha"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey!']

def conversation_chat(query, chain, history):
    result = chain({"question":query, "chat_history":history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF ")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_chain(vector_store):
    llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.1',
    model_kwargs = {"temperature": 0.7}
)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter=text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-l6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

vectorstore = FAISS.from_documents(splitter, embedding = embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory
)
return conversation_chain


def main():
    initialize_session_state()
    st.title("PDF Chatbot Using Mistral AI and AllMiniLM")
