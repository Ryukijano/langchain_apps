# streamlit_app.py
import streamlit as st
import uuid

st.title("Reasoning Agent")

session_id = st.session_state.get('session_id', str(uuid.uuid4()))
st.session_state['session_id'] = session_id

model_name = st.selectbox("Choose a model:", ["openai", "gemini"])
user_input = st.text_input("You:", key='input')

if st.button("Send"):
    response = cached_reasoning_agent(user_input, model_name, session_id)
    st.write(f"AI: {response}")
