from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class StaticListRetriever(BaseRetriever):
    """A retriever that returns a fixed list of documents, ignoring the query."""
    docs_list: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs_list

def getrelevantdocs(question, vector_db, docs_count=5, fetch_count=10):
    retrieved_docs = vector_db.max_marginal_relevance_search(question, k=docs_count, fetch_k=fetch_count)
    return retrieved_docs

def getuniquedocs(docs):
    seen_content = set()
    unique_chunks = []
    
    for chunk in docs:
        # Clean the string slightly (strip whitespace) to ensure accurate matching
        text_content = chunk.page_content.strip()
        
        if text_content not in seen_content:
            seen_content.add(text_content)
            unique_chunks.append(chunk)
            
    return unique_chunks
