from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def get_answer(model_name,vector_db, question):
    llm = OllamaLLM(model=model_name)
    
    template = """You are a job recommendation assistant for the Pakistani job market.
    Use ONLY the job listings provided below to answer the candidate's query.
    If no relevant jobs are found, say "No matching jobs found".

    For each relevant job, mention:
    - Job Title and Company
    - Location
    - Salary (if available)
    - Required Skills (briefly)
    - Apply Link: copy the EXACT Apply Link from the content else no link available".
    

    IMPORTANT: Do not make your own links if they arent available

    

    {context}

    Candidate Query: {question}

    Job Recommendations:"""
    
    prompt = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = qa_chain({"query": question})
    return result["result"]