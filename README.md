# Weather & Health Chatbot

A Streamlit-based chatbot that provides general weather-related health advice.  
It uses local text documents as a knowledge base, FAISS vector search for semantic retrieval,  
and Claude 3 Sonnet (via AWS Bedrock) to generate context-aware responses.

---

## Features
- Uses custom weather-health text documents as context
- Embeds and retrieves information using SentenceTransformer + FAISS
- Generates helpful responses using Claude 3 Sonnet
- Interactive chat interface built with Streamlit
- Provides general advice (non-medical)

---

## Tech Stack
- Python
- Streamlit
- SentenceTransformer (all-MiniLM-L6-v2)
- FAISS (Vector Similarity Search)
- AWS Bedrock — Claude 3 Sonnet

---

## Run Locally

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py


---

##  Disclaimer
This chatbot provides **general wellness guidance only**.  
It does **not** offer medical diagnosis.  
For serious health issues, consult a healthcare professional.

---
