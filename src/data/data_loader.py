# src/data_loader.py
from langchain_core.documents import Document
import pandas as pd

def load_jobs(csv_path: str) -> list[Document]:

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Title', 'Job Location'])
    df = df.drop_duplicates()

    docs = []
    for _, row in df.iterrows():
        content = f"""
        Job Title: {row['Title']}
        Location: {row['Job Location']}
        Job Type: {row['Job Type']}
        Salary: {row['Salary']}
        Skills Required: {row['Skills']}
        Career Level: {row.get('Career Level', 'Not specified')}
        Minimum Experience: {row.get('Minimum Experience', 'Not specified')}
        Minimum Education: {row.get('Minimum Education', 'Not specified')}
        Functional Area: {row.get('Functional Area', 'Not specified')}
        Apply Link: {row.get('Apply Link', 'Not specified')}
        """

        metadata = {
            "title": row['Title'],
            "location": row['Job Location'],
            "job_type": row['Job Type'],
            "salary": row['Salary'],
            "career_level": row.get('Career Level', 'Not specified'),
            "min_experience": row.get('Minimum Experience', 'Not specified'),
            "apply_link": row.get('Apply Link', 'Not specified'),
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs