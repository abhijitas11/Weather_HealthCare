import os
import faiss
import numpy as np
import json
import boto3
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------
# AWS Bedrock Setup
# -----------------------------
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",  # update if needed
    aws_access_key_id="abc",#paste your aws access key 
    aws_secret_access_key="abc")#paste your aws secret access key

# -----------------------------
# Load Local Documents
# -----------------------------
folder = r"D:\Documents\MCA-2\GenAI\WeatherHealthChatbot\weather_health_docs"
documents = []
for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# -----------------------------
# Create Embeddings and FAISS index
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -----------------------------
# Define System Prompt
# -----------------------------
system_prompt = (
    "You are a friendly and knowledgeable Weather & Health assistant. "
    "You answer clearly based on the given context about humidity, colds, coughs, breathing, "
    "temperature, and weather-related health advice. "
    "Encourage hydration and medical consultation if symptoms persist. "
    "Do not give medical diagnoses."
)

# -----------------------------
# Function to retrieve top docs
# -----------------------------
def retrieve_relevant_doc(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

# -----------------------------
# Function to call Claude
# -----------------------------
def ask_claude(question):
    top_docs = "\n\n".join(retrieve_relevant_doc(question))
    full_prompt = f"Context Document:\n{top_docs}\n\nUser Question: {question}"

    body = {
        "system": system_prompt,
        "messages": [{"role": "user", "content": full_prompt}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": 0.7
    }

    response = client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read().decode("utf-8"))
    return result["content"][0]["text"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🌤 Weather & Health Chatbot", layout="centered")
st.title("🌤 Weather & Health Chatbot")
st.markdown("Ask me questions about weather and health!")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask your weather-health question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = ask_claude(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error: {e}")
