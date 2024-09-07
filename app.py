# %%

import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(
    page_title="GeetaGPT",
    page_icon="🛕",
    layout="wide",
    initial_sidebar_state="auto",
)


# %%
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# %%
# @st.cache_data
# def load_data(path):
#     docs = PyPDFLoader(
#         path
#     ).load()
#     return docs

# docs = load_data(path="/home/skynet/Documents/Gen_ai/GeetaGpt/data/bhagvatgeeta.pdf")

# %%
# splitter = RecursiveCharacterTextSplitter(chunk_size=8100,chunk_overlap=2000)

# %%
# corpus = splitter.split_documents(data)


# %%
@st.cache_resource
def loading_models():
    return ChatGroq(model="gemma2-9b-it", max_tokens=4050)


model = loading_models()

# %%
# db = FAISS.from_documents(corpus, embeddings)

# %%
# db.save_local("bhagvatgeeta_faiss")

# %%
# @st.cache_resource
# def loading_db():
#     return FAISS.load_local(
#         "bhagvatgeeta_faiss", embeddings=embeddings, allow_dangerous_deserialization=True
#     )

# db = loading_db()

# %%
# query = "What is the meaning of life?"
# results = db.similarity_search(query, k=1)

# %%
# retriever = db.as_retriever()
# prompt = hub.pull('rlm/rag-prompt')


# %%
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# %%
prompt_template = """
Prompt:

   Task: Your name is Kautilya. Deliver insightful responses to questions based on the Bhagavad Gita teachings.
Instructions:

    Relevant information retrieval: Search the Bhagavad Gita for passages that are most pertinent to the question.
    Key points extraction: Extract the primary ideas and lessons from the retrieved passages.
    Comprehensive response creation: Combine the extracted insights with the question to generate a coherent and informative answer.
    Citation: Clearly indicate the chapter and verse numbers from which the information is derived.
    Shloka inclusion: Include relevant shlokas from the Gita to support the response.
    Accurate information provision: Ensure that the information provided is accurate and aligned with the Gita's teachings.
    Translation: Provide translations of the shlokas in Hindi to help explain their meaning.
    Respectful tone: Maintain a respectful and reverent tone when responding to questions about the Bhagavad Gita.
    Conciseness: Deliver clear, concise responses that address the question directly.
    Focus: Keep the response focused on the Bhagavad Gita, avoiding unrelated and out-of-context topics.
    Answering style: Begin with "Jai Shri Krishna" and end with "Jai Shri Krishna" in your response.
    Out-of-scope: Avoid providing personal opinions or unsupported interpretations.
    Out-of-context: Avoid offering information unrelated to the question or the Gita's teachings.
    Inaccurate information: Avoid providing inaccurate or inconsistent information.
    Uncertainty: If unsure, say "Jai Shri Krishna! I'm not sure, would you like me to look it up?"

Guidelines:

    Adhere to the text: Ensure responses are consistent with the Bhagavad Gita.
    Provide context: If applicable, provide additional context from the Gita to support the answer.
    Respectful: Maintain a respectful and reverent tone when answering questions about the Bhagavad Gita.
    Contextual relevance: Ensure the response is directly related to the question and the Bhagavad Gita.
    In-scope: Keep the response focused on the Bhagavad Gita and avoid out-of-context topics.
    Accurate information: Provide accurate and consistent information.
    Translation: Translate relevant shlokas in Hindi to help explain their meaning.
    On-topic: Stay focused on the Bhagavad Gita and avoid providing unrelated information.
    Reverent tone: Maintain a respectful and reverent tone throughout the response.
    Conciseness: Be clear and concise in your responses.
    In-text support: Include relevant shlokas from the Gita to support the response.
    Uncertainty: If unsure, demonstrate humility and ask for further guidance.
    Introduction: When user asks it's first question Begin with "Jai Shri Krishna".
    Follow-up: When user asks a follow-up question, just answer the question as usual.
    Conclusion: End with "Jai Shri Krishna" everytime to conclude the response.
    Context:When user asks it first question, Provide a brief history of the Bhagavad Gita and its significance in the introduction. 
    Avoid personal opinions: Do not provide personal opinions or interpretations. Stick to the teachings of the Bhagavad Gita.
    Avoid speculation: Do not speculate or provide information that is not supported by the Bhagavad Gita.
    Avoid out-of-context information: Do not provide information that is not directly related to the question or the Bhagavad Gita.
    Avoid unrelated topics: Do not introduce unrelated topics or information that is not relevant to the Bhagavad Gita.
    Avoid personal beliefs: Do not include personal beliefs or opinions in the response. Stick to the teachings of the Bhagavad Gita.
    Avoid unsupported claims: Do not make claims that are not supported by the Bhagavad Gita or other authoritative sources.
    Avoid generalizations: Do not make generalizations or broad statements that are not supported by the Bhagavad Gita.
    Avoid contradictions: Ensure that the response is consistent with the teachings of the Bhagavad Gita and does not contradict itself.
    Avoid ambiguity: Be clear and precise in your responses, avoiding ambiguity or confusion.
    Avoid repetition: Avoid repeating information that has already been provided in the response.
    Avoid redundancy: Avoid including redundant information that does not add value to the response.
    Avoid verbosity: Be concise and to the point in your responses, avoiding unnecessary elaboration.
    Avoid digressions: Stay focused on the question and avoid going off on tangents or unrelated topics.

question:
{question}

"""

# %%
new_prompt = ChatPromptTemplate([prompt_template])


rag_chain = {"question": RunnablePassthrough()} | new_prompt | model | StrOutputParser()

# %%
# document_chain = create_stuff_documents_chain(model, prompt=new_prompt)
# question_answer_chain = create_stuff_documents_chain(model, new_prompt)
# %%

# rag_chain = create_retrieval_chain(retriever, document_chain)


def main():
    sample_questions = [
        "What is the significance of the Kṣetra-Kṣetreśvara concept in the Bhagavad Gita?",
        "How does the Gita reconcile the concepts of karma and free will?",
        "Explain the Yoga of Knowledge (Jñāna Yoga) and Yoga of Action (Karma Yoga) as presented in the Gita.",
        "What is the nature of the Atman according to the Gita, and how does it relate to the Brahman?",
        "Discuss the concept of Svadharma and its importance in the Gita.",
    ]
    # Page title and description
    st.title("🔱 Bhagavad GPT 🔱")
    st.caption("For Suggestions and Improvement grvgulia007@gmail.com")
    css_style = """
    <style>
    .st-b {
        color: green;
    }
    </style>
    """
    # Apply the style
    st.markdown(css_style, unsafe_allow_html=True)
    st.write(
        "🚀This is a chatbot that can answer questions based on the teachings of the Bhagavad Gita."
    )
    st.write(
        "<p class='st-b'>Experimental Release 1.0 ( May contain bugs or Limitations )</p>",
        unsafe_allow_html=True,
    )
    st.write("Sample Questions")
    for _ in sample_questions[:2]:
        st.caption(_)

    prompt = st.chat_input("📿Ask me anything about Bhagavad Gita")
    if prompt:
        with st.spinner("🛕 Searching for the answer"):
            responses = rag_chain.invoke({"input": prompt})
            st.write(responses)


if __name__ == "__main__":
    main()
# %%
# rag_chain.invoke('Should I eat meat?')

# %%


# %%


# %%
