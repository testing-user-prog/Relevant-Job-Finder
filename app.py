from src.data.data_loader import load_jobs
from src.features.vectorizer import makevectordb
from src.generation.responder import get_answer
from src.data.csv_maker import makejobpostingcsv
import json
import os
question="I am looking for the internships that I can do remotely"
config=None
with open("config.json", "r") as f:
    config = json.load(f)
#data loading, cleaning and other stuff->after this we get like the document objects
makejobpostingcsv(config['csv_path'],config['search_keyword'],config['jsearch_api'],config['csv_name'],config['ollama_model_csvmaker'],config['groq_api_key'])
docs=load_jobs(os.path.join(config['csv_path'], config['csv_name']))
vector_db=makevectordb(docs,config['Huggingface_embedder_model'])
response=get_answer(config['ollama_model_responder'],vector_db,question,config['groq_api_key'])
print(response)
print('Program working fine')
