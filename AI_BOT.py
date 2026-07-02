import streamlit as st
import faiss
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
DistilBertTokenizerFast,
DistilBertForSequenceClassification,
pipeline
)

from huggingface_hub import InferenceClient



# CONFIG



HF_TOKEN = "___hugging&face_token___"



# LOAD EVERYTHING



@st.cache_resource
def load_resources():


    tokenizer = DistilBertTokenizerFast.from_pretrained(
        r"D:\obito\obito\final\intent_model"
    )

    intent_model = DistilBertForSequenceClassification.from_pretrained(
        r"D:\obito\obito\final\intent_model"
    )

    with open(
        r"D:\obito\obito\final\label_encoder.pkl",
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        r"D:\obito\obito\final\zends_faiss.index"
    )

    with open(
        r"D:\obito\obito\final\chunks.pkl","rb"
    ) as f:

        chunks = pickle.load(f)

    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN
    )

    return (
        tokenizer,
        intent_model,
        label_encoder,
        sentiment_model,
        embed_model,
        index,
        chunks,
        client
    )
(
tokenizer,
intent_model,
label_encoder,
sentiment_model,
embed_model,
index,
chunks,
client
) = load_resources()



# INTENT



def predict_intent(query):

    inputs = tokenizer(
         query,
         return_tensors="pt",
         truncation=True,
         padding=True
         )

    with torch.no_grad():

         outputs = intent_model(**inputs)

    pred = outputs.logits.argmax(dim=1).item()

    return label_encoder.inverse_transform(
                    [pred]
                )[0]




# SENTIMENT



def predict_sentiment(query):


    result = sentiment_model(
        query
    )

    return result[0]["label"]




# RETRIEVE CONTEXT



def retrieve_context(query):


    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    D, I = index.search(
        query_embedding.astype("float32"),
        k=1
    )

    context = "\n".join(
        [chunks[idx] for idx in I[0]]
    )

    return context[:1500]




# CHATBOT



def chatbot(query):
    # Predict intent
    intent = predict_intent(query)

    # Predict sentiment
    sentiment = predict_sentiment(query)

    # Retrieve context
    context = retrieve_context(query)

    # Build prompt
    prompt = f"""
You are a professional customer support assistant for ZENDS Communications.

Intent: {intent}
Sentiment: {sentiment}

Company Information:
{context}

Customer Question:
{query}

Instructions:
- Respond in no more than 2 sentences.
- Keep the response under 40 words.
- Finish the response completely.
- Do not end the response mid-sentence.
- Do not ask multiple follow-up questions.
"""

    # Generate response
    messages = [
        {
            "role": "system",
            "content": "You are a professional customer support assistant for ZENDS Communications."
        }
    ]

    for msg in st.session_state.messages:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=messages,
        max_tokens=100,
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return intent, sentiment, answer




# STREAMLIT UI



st.set_page_config(page_title="AI Customer Support Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 AI Customer Support Copilot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Type your message...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        intent, sentiment, answer = chatbot(query)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)

    
    
