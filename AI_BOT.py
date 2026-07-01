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

# ==========================

# CONFIG

# ==========================

HF_TOKEN = "___hugging_face_token___"

# ==========================

# LOAD EVERYTHING

# ==========================

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

# ==========================

# INTENT

# ==========================

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


# ==========================

# SENTIMENT

# ==========================

def predict_sentiment(query):


    result = sentiment_model(
        query
    )

    return result[0]["label"]


# ==========================

# RETRIEVE CONTEXT

# ==========================

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


# ==========================

# CHATBOT

# ==========================

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

Give a short and professional answer.
"""

    # Generate response
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=64
    )

    answer = response.choices[0].message.content

    return intent, sentiment, answer


# ==========================

# STREAMLIT UI

# ==========================

st.set_page_config(
page_title="AI Customer Support Copilot"
)

st.title(
"🤖 AI Customer Support Copilot"
)

query = st.text_area(
"Enter Customer Query"
)

if st.button("Generate Response"):

    if query.strip() == "":
        st.warning("Please enter a query.")

    else:

        with st.spinner("Processing..."):

            intent, sentiment, answer = chatbot(query)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Intent")
            st.success(intent)

        with col2:
            st.subheader("Sentiment")
            st.info(sentiment)

        st.subheader("Bot")
        st.write(answer)

    
    
