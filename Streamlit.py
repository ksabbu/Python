import streamlit as st
from alert_query_bot import AlertQueryBot

# Initialize bot
bot = AlertQueryBot()

# Streamlit UI
st.set_page_config(page_title="Alert Query Bot", layout="centered")
st.title("📢 Alert Query Chatbot")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask about an alert...")
if user_query:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant context
    relevant_text = bot.retrieve_relevant_text(user_query)

    # Query the LLM
    bot_response = bot.query_bot(user_query, relevant_text)

    # Display bot response
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
