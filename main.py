import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

st.markdown("jai baba ki ladle")
st.markdown("k keh tha gaade rokega")
# Load API key from .env
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-5.4-mini-2026-03-17",
    temperature=0.7
)

# Page title
st.title("🎓 Success Coach AI")

# Create message history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at bottom
prompt = st.chat_input("Ask me anything...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Generate response
    response = llm.invoke(prompt)
    answer = response.content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )