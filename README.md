# Relevant Job Finder 🇵🇰

A RAG (Retrieval-Augmented Generation) based job recommendation system for the Pakistani job market. It fetches fresh job postings from the web, builds a semantic search index, and uses a cloud LLM to recommend the most relevant jobs based on your query — in plain English.

---

## How It Works

```
Search Keyword → JSearch API → Fresh Jobs CSV
                                      ↓
                              Data Loader (LangChain Documents)
                                      ↓
                         HuggingFace Embeddings + ChromaDB (in-memory)
                                      ↓
                     MMR Retrieval → Deduplication → Custom Retriever
                                      ↓
                        User Query → RAG Pipeline → Groq LLM
                                      ↓
                              Job Recommendations (Streamlit UI)
```

1. **CSV Maker** — fetches live job postings from JSearch API based on a search keyword, uses Groq LLM to extract structured fields from job descriptions, and saves them as a CSV
2. **Data Loader** — reads the CSV, cleans it, and converts each job into a LangChain `Document` object
3. **Vectorizer** — embeds all job documents using HuggingFace (`all-MiniLM-L6-v2`) and stores them in an in-memory ChromaDB vectorstore
4. **Retriever** — uses MMR (Maximal Marginal Relevance) search to fetch diverse relevant documents, deduplicates them, and wraps them in a custom `StaticListRetriever`
5. **Responder** — passes retrieved jobs to Groq LLM via a custom prompt and generates structured job recommendations
6. **Streamlit UI** — a clean web interface where users enter their job preferences and get recommendations instantly

---

## Project Structure

```
Relevant-Job-Finder/
│
├── src/
│   ├── data/
│   │   ├── csv_maker.py        # Fetches jobs from JSearch API → saves CSV
│   │   └── data_loader.py      # Loads CSV → LangChain Document objects
│   │
│   ├── features/
│   │   ├── vectorizer.py       # Embeds documents → in-memory ChromaDB
│   │   └── retriever.py        # MMR search → deduplication → StaticListRetriever
│   │
│   └── generation/
│       └── responder.py        # RAG chain → Groq LLM → job recommendations
│
├── dataset/                    # Auto-created, stores fetched jobs CSV
├── app.py                      # Streamlit entry point
├── config.json                 # All configuration in one place
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/relevant-job-finder.git
cd relevant-job-finder
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a JSearch API key
Sign up for free at [rapidapi.com](https://rapidapi.com) and subscribe to the **JSearch** API (free tier). Copy your API key.

### 5. Get a Groq API key
Sign up for free at [console.groq.com](https://console.groq.com) and generate an API key.

### 6. Configure `config.json`
```json
{
    "groq_model_csvmaker": "llama-3.1-8b-instant",
    "groq_model_responder": "llama-3.3-70b-versatile",
    "search_keyword": "AI/ML Internships",
    "csv_path": "dataset",
    "csv_name": "fetchedjobs.csv",
    "jsearch_api": "<your JSearch API key here>",
    "Huggingface_embedder_model": "all-MiniLM-L6-v2",
    "groq_api_key": "<your Groq API key here>"
}
```

---

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Enter your job preferences in the text box (e.g. *"I am looking for remote AI internships"*)
2. Click **Find Jobs**
3. The system fetches fresh jobs, builds a semantic index, and returns personalized recommendations with apply links

---

## Configuration Options

| Key | Description |
|-----|-------------|
| `groq_model_csvmaker` | Groq model used to extract job fields from descriptions |
| `groq_model_responder` | Groq model used to generate job recommendations |
| `search_keyword` | Keyword sent to JSearch API to fetch jobs |
| `csv_path` | Folder where the fetched jobs CSV is saved |
| `csv_name` | Name of the generated CSV file |
| `jsearch_api` | Your RapidAPI key for JSearch |
| `Huggingface_embedder_model` | HuggingFace sentence-transformer model for embeddings |
| `groq_api_key` | Your Groq API key for LLM inference |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Job Data | JSearch API (via RapidAPI) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| Retrieval | MMR Search + Custom `StaticListRetriever` |
| LLM | Groq (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) |
| RAG Framework | LangChain |
| Data Processing | Pandas |
| Frontend | Streamlit |

---

## Requirements

```
langchain
langchain-community
langchain-classic
langchain-ollama
langchain-huggingface
langchain-core
langchain-groq
chromadb
sentence-transformers
ollama
requests
pandas
streamlit
```

---

## Notes

- The vectorstore is **in-memory only** — it rebuilds fresh on every run with the latest job data
- The `dataset/` folder is auto-created if it doesn't exist
- JSearch free tier allows up to 200 requests/month — more than enough for personal use
- Groq free tier is sufficient for personal and portfolio use
- Add `config.json` to `.gitignore` to avoid exposing your API keys — use `config.example.json` as a template instead