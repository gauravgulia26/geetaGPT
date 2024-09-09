import os
import time
import pyttsx3
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import streamlit as st

st.set_page_config(layout="wide")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from streamlit_extras.streaming_write import write

# Load environment variables (assuming they are set)
load_dotenv()


# Function to load HuggingFace embeddings (cached)
@st.cache_data()
def loading_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load embeddings (use session state for persistence)
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = loading_embeddings()
embeddings = st.session_state["embeddings"]


# Function to load FAISS database (cached)
@st.cache_resource()
def loading_db():
    return FAISS.load_local(
        "/home/skynet/Documents/Gen_ai/GeetaGpt/notebooks/bhagvatgeeta_new",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


# Load FAISS database (use session state)
if "db" not in st.session_state:
    st.session_state["db"] = loading_db()
db = st.session_state["db"]

# Create retriever (no need to cache, it's stateless)
retriever = db.as_retriever()


# Function to load ChatGroq model (cached)
@st.cache_resource()
def loading_models():
    return ChatGroq(model="gemma2-9b-it", max_tokens=4050)


# Load ChatGroq model (use session state)
if "model" not in st.session_state:
    st.session_state["model"] = loading_models()
model = st.session_state["model"]

prompt = ChatPromptTemplate.from_template(
    """
    Task: Answer questions based on the Bhagavad Gita teachings in Hindi, incorporating the provided context.

    Instructions:

        Retrieve relevant information: Search the Bhagavad Gita for passages that are most relevant to the query, considering both the query itself and the provided context.
        Summarize key points: Extract the main ideas and concepts from the retrieved passages, taking into account the context for a more nuanced understanding.
        Generate a comprehensive response: Combine the summarized information with the query and context to create a coherent and informative answer in Hindi.
        Mention references: Clearly indicate the chapter and verse numbers from which the information is extracted.
        Mention Sholkas: Clearly include relevant shlokas from the Gita to support your response.
        Provide accurate information: Ensure that the information provided is accurate and consistent with the teachings of the Gita, considering the context.
        Stay focused: Keep your response focused on the query, considering both the Gita's teachings and the provided context.
        Be concise: Provide a clear and concise response in Hindi that addresses the query effectively.
        Include examples: Use examples from the Gita to illustrate key concepts and teachings.
        Introduction: If a user says "Hello" or "Hi", respond with a greeting and an introduction to the chatbot in Hindi.
        Final Response: Your final response should be a well-structured and informative answer that addresses the query effectively and the language should be in Hindi strictly.

    Guidelines:

        Stay true to the text: Ensure that your responses are consistent with the teachings of the Bhagavad Gita.
        Provide context: If applicable, provide additional context from the Gita to support your answer, considering the provided context.
        Be respectful: Maintain a respectful and reverent tone when answering questions about the Bhagavad Gita.
        Be concise: Provide clear and concise responses in Hindi that address the query directly.
        Stay focused: Ensure that your response is directly related to the Bhagavad Gita and avoids unrelated and out-of-context topics.
        Answering Style: Start with "हे पार्थ" and end with "जय श्री कृष्ण" in your answer.
        Out of scope: Avoid providing personal opinions or interpretations that are not supported by the text of the Gita.
        Out of Context: Avoid providing information that is not directly related to the query or the teachings of the Gita, considering the provided context.
        Inaccurate Information: Avoid providing information that is inaccurate or inconsistent with the teachings of the Gita.
        Not Sure: If you don't know the answer: Say "मैं इस प्रश्न का उत्तर अभी नहीं दे पा रहा हूँ। कृपया एक और प्रश्न पूछें। जय श्री कृष्ण!"

    <context>
    {context}
    <context>
    Question:{input}

    """
)

# Create retrieval chain (no need to cache)
chain = create_stuff_documents_chain(model, prompt)  # Placeholder for prompt
retrieval_chain = create_retrieval_chain(retriever, chain)

# Sample questions (can be moved to a separate file)
sample_questions = [
    "What is the significance of the Kṣetra-Kṣetreśvara concept in the Bhagavad Gita?",
    "How does the Gita reconcile the concepts of karma and free will?",
    "Explain the Yoga of Knowledge (Jñāna Yoga) and Yoga of Action (Karma Yoga) as presented in the Gita.",
    "What is the nature of the Atman according to the Gita, and how does it relate to the Brahman?",
    "Discuss the concept of Svadharma and its importance in the Gita.",
]


def main():
    with st.sidebar:
        st.title(" Bhagavad GPT ")
        st.markdown(
            "This is a ChatBot that can answer questions based on the teachings of the Bhagavad Gita."
        )
        st.write("Sample Questions")
        for _ in sample_questions[:2]:
            st.caption(_)

        st.link_button("Contact Us", "mailto:grvgulia007@gmail.com")

    # Use session state to store the user prompt
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ""

    prompt = st.chat_input("Ask me anything about Bhagavad Gita", key="user_prompt")

    # Update session state only if the prompt has changed
    if prompt != st.session_state["prompt"]:
        st.session_state["prompt"] = prompt
        with st.spinner(" Searching for the answer. Please Wait !"):
            response = retrieval_chain.invoke({"input": prompt})["answer"]
            st.write(response)
        # Optionally, play the response using TTS (implementation not included)
