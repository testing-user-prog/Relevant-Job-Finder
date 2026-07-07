from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def get_answer(model_name, question, groq_api, retriever):
    llm = ChatGroq(model=model_name, api_key=groq_api)
    
    template = """You are a job recommendation assistant for the Pakistani job market.
    Use ONLY the job listings provided below to answer the candidate's query.
    If no relevant jobs are found, say "Unable to find the job".

    For each relevant job, display exactly the following format as a bulleted list. Output NOTHING ELSE besides these fields for each job:
    - **Job Title:** <title>
    - **Company:** <company>
    - **Location:** <location>
    - **Salary:** <salary if available, else "Not specified">
    - **Required Skills:** <skills briefly>
    - **Apply Link:** <EXACT link from content, else "No link available">

    STRICT RULES:
    - FILTER STRICTLY: You must evaluate each job against the Candidate Query. If a job does not closely match the user's requirements (e.g. skills, role, location), you MUST completely exclude it from your output.
    - If all jobs are excluded after filtering, output "Unable to find the job" and nothing else.
    - You MUST use the bullet points as shown above so it formats correctly.
    - Separate multiple job listings with a single blank line.
    - Do NOT add any introductory text before the listings.
    - Do NOT add any notes, disclaimers, warnings, or closing remarks after the listings.
    - Do NOT add any text after the last job listing. Stop immediately after the final Apply Link.
    - Do NOT invent or modify any information not present in the context.

    {context}
 
    Candidate Query: {question}

    Job Recommendations:"""
    
    prompt = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = qa_chain({"query": question})
    return result["result"]

