from data.employees import generate_employee_data
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq
from gui import AssistantGUI


if __name__ == "__main__":

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Umbrella Onboarding", page_icon="☂️", layout="wide")


    @st.cache_data(ttl=3600, show_spinner="Loading Employee data...")
    def get_user_data():
        return generate_employee_data(1)[0]
    
    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            persistent_path = "./data/vectorstore"

            vectorstore = Chroma.from_documents(
                documents = splits,
                embedding = embedding_function,
                persist_directory = persistent_path,
            )

            return vectorstore
        except Exception as e:
            logging.error(f"Error Initalizing vector store: {str(e)}")
            st.error(f"Failed to initalize vector store: {str(e)}")
            return None


    customer_data = get_user_data()
    vector_store = init_vector_store("data/umbrella_corp_policies.pdf")

    if vector_store is None:
        st.error("Vector store could not be initialized. Please check the logs for more details.")
        st.stop()

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    llm = ChatGroq(model="llama-3.1-8b-instant")

    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        employee_information=st.session_state.customer,
        vector_store=vector_store,
    )

    gui = AssistantGUI(assistant)
    gui.render()