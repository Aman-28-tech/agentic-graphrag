from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role":"user","content":"What is BERT?"}
    ]
)

print(response.choices[0].message.content)