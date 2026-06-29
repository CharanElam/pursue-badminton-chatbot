import os
import streamlit as st
from rag import load_rag_chain

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="Badminton Academy Chatbot",
    page_icon="🏸",
    layout="centered"
)

st.title("🏸 Badminton Academy Assistant")
st.caption("Ask me anything about our academy!")

@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})