from src.data.data_loader import load_jobs
from src.features.vectorizer import makevectordb
from src.generation.responder import get_answer
from src.features.retriever import getrelevantdocs, getuniquedocs, StaticListRetriever
from src.data.csv_maker import makejobpostingcsv
import json
import os
import streamlit as st

question = ""

st.markdown("<h1 style='text-align:center; margin-bottom:10px;'>Relevant Job Finder</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 2px solid #2e86c1 !important;
        padding: 10px !important;
        width: 100% !important;
    }
    .stButton > button {
        background-color: #2e86c1 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0px
        width: 100% !important;
        white-space: nowrap !important;
    }
    .stButton > button:hover {
        background-color: #ff9999 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
config=None
with open("config.json", "r") as f:
    config = json.load(f)
col1, col2, col3 = st.columns([1, 2, 1])
def printresults(result):
        st.write(result)
with col2:
    def jobfetchingprocess():
        with st.spinner("Fetching the best jobs for you..."):
            makejobpostingcsv(config['csv_path'],config['search_keyword'],config['jsearch_api'],config['csv_name'],config['groq_model_csvmaker'],config['groq_api_key'])
            docs=load_jobs(os.path.join(config['csv_path'], config['csv_name']))
            vector_db=makevectordb(docs,config['Huggingface_embedder_model'])
            relevant_docs=getrelevantdocs(question,vector_db)
            unique_docs=getuniquedocs(relevant_docs)
            custom_retriever=StaticListRetriever(docs_list=unique_docs)
            response=get_answer(config['groq_model_responder'],question,config['groq_api_key'],custom_retriever)
            return response
    

    question = st.text_input("Enter Job details")
    if "emptyflag" not in st.session_state:
        st.session_state.emptyflag = False
    if "answer" not in st.session_state:
        st.session_state.answer = None
    sub1, sub2, sub3 = st.columns([1, 1, 1])
    with sub2:
        if st.button("Find Jobs"):
            if question == "":
                st.session_state.emptyflag = True
            else:
                print(question)
                st.session_state.emptyflag = False
                #data loading, cleaning and other stuff->after this we get like the document objects
                st.session_state.answer = jobfetchingprocess()

    if st.session_state.emptyflag == True:
        st.session_state.emptyflag = False
        st.error("Please type some details")

if st.session_state.get("answer"):
    printresults(st.session_state.answer)
        