import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel,ValidationError

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

class JobAnalysis(BaseModel):
    job_title: str
    required_skills: list[str]
    preferred_skills: list[str]
    experience_level: Literal["intern", "junior", "mid", "senior"]
    summary: str

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

user_input = input("Job Description: ")

messages= [
    {
        "role": "system",
        "content": """
        You are a job description information extraction system.

Your task is to extract information ONLY from the provided job description.

Rules:
1. Do not infer, guess, or invent information.
2. Extract every explicitly mentioned technology.
3. If a technology is described as something the candidate should know,
   have, use, or have experience with, include it in required_skills.
4. Only put skills in preferred_skills when the job description explicitly
   describes them as preferred, optional, nice-to-have, or a plus.
5. Words that are only part of the job title must not automatically be
   treated as skills.
6. If required vs preferred is not explicitly distinguished, treat
   explicitly mentioned technologies as required.
7. experience_level must be exactly one of:
   "intern", "junior", "mid", "senior".

Return valid JSON with exactly these fields:
job_title
required_skills
preferred_skills
experience_level
summary

Example:

Job description:
"We are looking for a backend intern with Python, SQL, Git and Docker experience."

Output:
{
  "job_title": "Backend Intern",
  "required_skills": ["Python", "SQL", "Git", "Docker"],
  "preferred_skills": [],
  "experience_level": "intern",
  "summary": "The employer is looking for a backend intern with Python, SQL, Git and Docker experience."
}

The JSON must contain:

- job_title:
  The job title explicitly stated or clearly identifiable from the job description.

- required_skills:
  Skills explicitly described as required.

- preferred_skills:
  Skills explicitly described as preferred or optional.
  If no preferred skills are explicitly mentioned, return an empty list.

- experience_level:
  MUST be exactly one of these four strings:
  "intern", "junior", "mid", "senior"

  Do not use values such as "internship", "entry-level",
  "graduate", or any other variation.

- summary:
  A short summary based only on the provided job description.
  Do not add information that is not present in the job description.



        """
    },

        {
            "role": "user",
            "content": f"""
Extract information from the following job description:

<job_description>
{user_input}
</job_description>
"""
        }

]

response = client.chat.completions.create(
    model = "deepseek-flash",
    messages=messages,
    response_format= {
        "type": "json_object"
    },
    stream=False
)

ai_reply = response.choices[0].message.content
data= json.loads(ai_reply)
job= JobAnalysis.model_validate(data)
print(job.job_title)
print(job.required_skills)
print(job.preferred_skills)
print(job.experience_level)
print(job.summary)


