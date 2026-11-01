import streamlit as st
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# --- INTENTS DATA ---
intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "salam", "assalam o alaikum", "good morning"],
        "responses": ["Hello! 👋 How can I help you?", "Hi there! Kaisy madad kar sakta hun?"]
    },
    "farewells": {
        "patterns": ["bye", "goodbye", "allah hafiz", "khuda hafiz", "see you"],
        "responses": ["Allah Hafiz! 👋 Take care", "Bye! Phir milenge"]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "tum kon ho", "apka naam"],
        "responses": ["I am Simple AI Chatbot, made with Python + NLTK + TF-IDF"]
    },
    "help": {
        "patterns": ["help me", "can you help", "madad chahiye", "what can you do"],
        "responses": ["I can chat! I understand greetings, FAQs, basic questions using NLP"]
    },
    "ai_info": {
        "patterns": ["what is ai", "what is artificial intelligence", "ai kya hai", "nlp kya hai"],
        "responses": ["AI is Artificial Intelligence - machines that think like humans. NLP is how machines understand human language."]
    },
    "project_info": {
        "patterns": ["how you work", "how do you work", "tf idf", "cosine similarity"],
        "responses": ["I use NLTK Tokenizer + WordNet Lemmatizer to normalize text, then TF-IDF + Cosine Similarity to match intent!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "shukria", "jazakallah"],
        "responses": ["Welcome! 😊", "Khush Amdeed!"]
    }
}

all_patterns = []
all_tags = []
for tag, data in intents.items():
    for pattern in data["patterns"]:
        all_patterns.append(pattern)
        all_tags.append(tag)

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

normalized_patterns = [normalize_text(p) for p in all_patterns]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(normalized_patterns)

def get_response(user_input):
    norm_input = normalize_text(user_input)
    if not norm_input.strip():
        return "Please likhiye kuch..."
    input_vec = vectorizer.transform([norm_input])
    similarities = cosine_similarity(input_vec, tfidf_matrix)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]
    if best_score < 0.2:
        return "Sorry, samajh nahi aaya 🤔. Try: hello, who are you, what is AI, help, bye"
    best_tag = all_tags[best_idx]
    import random
    return random.choice(intents[best_tag]["responses"]) + f"\n\n_[Intent: {best_tag} | Score: {best_score:.2f}]_"

st.set_page_config(page_title="Simple AI Chatbot", page_icon="🤖")
st.title("🤖 6. Simple AI Chatbot")
st.markdown("**Project Idea:** Rule-based chatbot in Python using NLTK + TF-IDF + Cosine Similarity")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salam! I am your AI Chatbot. Try: 'hello', 'who are you', 'what is AI', 'help'"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Yahan likho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)