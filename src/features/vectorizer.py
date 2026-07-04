from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def makevectordb(docs, embedding_model="all-MiniLM-L6-v2"):
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        
        

    )
    return db