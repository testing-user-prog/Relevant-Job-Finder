import requests
import pandas as pd
import os
import json
from groq import Groq

def extract_fields_from_description(description, model_name,groq_api):
    client=Groq(api_key=groq_api)
    prompt = f"""Extract the following fields from this job description.
Return ONLY a valid JSON object with these exact keys, nothing else:
{{
    "Functional Area": "department or area e.g. IT, Sales, Marketing",
    "Career Level": "e.g. Entry Level, Mid Level, Senior, Manager",
    "Minimum Experience": "e.g. 1 Year, 2 Years, Fresh",
    "Minimum Education": "e.g. Bachelor's, Master's, Intermediate",
    "Gender": "Male, Female, or No Preference",
    "Age": "age range if mentioned, otherwise null",

}}

Job Description:
{description[:1000]}

JSON:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    
    try:
        raw = response.choices[0].message.content
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except:
        return {
            "Functional Area": None,
            "Career Level": None,
            "Minimum Experience": None,
            "Minimum Education": None,
            "Gender": None,
            "Age": None
        }


def makejobpostingcsv(path, search_keyword, apikey, csvname, model_name,groq_api):
    url = "https://jsearch.p.rapidapi.com/search-v2"
    

    querystring = {
        "query": search_keyword,
        "country": "pk",
        "num_pages": "1"
    }

    headers = {
        "x-rapidapi-key": apikey,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    jobs = []
    for job in data["data"]["jobs"]:
        description = job.get("job_description", "")
        
        # extract missing fields using LLM
        extracted = extract_fields_from_description(description, model_name,groq_api)
        jobs.append({
        "Title":              job.get("job_title"),
        "Salary":             job.get("job_salary_string"),
        "Job Type":           job.get("job_employment_type"),
        "Job Location":       f"{job.get('job_city')}, {job.get('job_country')}",
        "Functional Area":    None,
        "Career Level":       None,
        "Apply Before":       job.get("job_posted_at"),
        "Minimum Experience": None,
        "Minimum Education":  None,
        "Gender":             None,
        "Age":                None,
        "Skills":             description[:300],
        "Apply Link":         job.get("job_apply_link"),  # ← add this
    })
        

    df = pd.DataFrame(jobs)
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, csvname)
    df.to_csv(output_path, index=False)

    print(f"\n{len(jobs)} jobs saved to {output_path}")
    return df