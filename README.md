# Relevant Job Finder 🇵🇰

A RAG (Retrieval-Augmented Generation) based job recommendation system for the Pakistani job market. It fetches fresh job postings from the web, builds a semantic search index, and uses a local LLM to recommend the most relevant jobs based on your query — in plain English.

---

## How It Works

```
Search Keyword → JSearch API → Fresh Jobs CSV
                                      ↓
                              Data Loader (LangChain Documents)
                                      ↓
                         HuggingFace Embeddings + ChromaDB (in-memory)
                                      ↓
                        User Query → RAG Pipeline → Ollama LLM
                                      ↓
                              Job Recommendations
```

1. **CSV Maker** — fetches live job postings from JSearch API based on a search keyword and saves them as a CSV
2. **Data Loader** — reads the CSV, cleans it, and converts each job into a LangChain `Document` object
3. **Vectorizer** — embeds all job documents using HuggingFace (`all-MiniLM-L6-v2`) and stores them in an in-memory ChromaDB vectorstore
4. **Responder** — takes the user's natural language query, retrieves the most relevant jobs from the vectorstore, and passes them to a local Ollama LLM to generate a personalized recommendation

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
│   │   └── vectorizer.py       # Embeds documents → in-memory ChromaDB
│   │
│   └── generation/
│       └── responder.py        # RAG chain → Ollama LLM → job recommendations
│
├── dataset/                    # Auto-created, stores fetched jobs CSV
├── app.py                      # Entry point
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

### 4. Install and start Ollama
Download from [ollama.com](https://ollama.com/download) then pull the required models:
```bash
ollama pull llama3.2:1b       # for CSV field extraction
ollama pull qwen2.5:7b        # for job recommendations
```

### 5. Get a JSearch API key
Sign up for free at [rapidapi.com](https://rapidapi.com) and subscribe to the **JSearch** API (free tier). Copy your API key.

### 6. Configure `config.json`
```json
{
    "ollama_model_csvmaker": "llama3.2:1b",
    "ollama_model_responder": "qwen2.5:7b",
    "search_keyword": "Computing Internships",
    "csv_path": "dataset",
    "csv_name": "fetchedjobs.csv",
    "jsearch_api": "<your JSearch API key here>",
    "Huggingface_embedder_model": "all-MiniLM-L6-v2"
}
```

---

## Usage

Edit the question in `app.py` to match what you're looking for:

```python
question = "I am looking for internships that I can do remotely"
```

Then run:
```bash
python app.py
```

The system will:
1. Fetch fresh job postings matching your `search_keyword` from JSearch
2. Build an in-memory semantic search index
3. Find the most relevant jobs for your query
4. Generate a recommendation with job title, location, salary, skills, and apply link

---

## Configuration Options

| Key | Description |
|-----|-------------|
| `ollama_model_csvmaker` | Ollama model used to extract job fields from descriptions |
| `ollama_model_responder` | Ollama model used to generate job recommendations |
| `search_keyword` | Keyword sent to JSearch API to fetch jobs |
| `csv_path` | Folder where the fetched jobs CSV is saved |
| `csv_name` | Name of the generated CSV file |
| `jsearch_api` | Your RapidAPI key for JSearch |
| `Huggingface_embedder_model` | HuggingFace sentence-transformer model for embeddings |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Job Data | JSearch API (via RapidAPI) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| LLM | Ollama (`llama3.2:1b`, `qwen2.5:7b`) |
| RAG Framework | LangChain |
| Data Processing | Pandas |

---

## Requirements

```
langchain
langchain-community
langchain-classic
langchain-ollama
langchain-huggingface
langchain-core
chromadb
sentence-transformers
ollama
requests
pandas
```

---

## Notes

- The vectorstore is **in-memory only** — it rebuilds fresh on every run with the latest job data
- Ollama must be **running in the background** before you run `app.py`
- The `dataset/` folder is auto-created if it doesn't exist
- JSearch free tier allows up to 200 requests/month — more than enough for personal use

---

## Future Improvements

- Streamlit UI for a cleaner user experience
- Swap Ollama for Groq API for cloud deployment
- Add conversational memory for multi-turn job search
- Support for filtering by location, salary range, and job type
- Deploy on Streamlit Cloud with Groq as the LLM backend
