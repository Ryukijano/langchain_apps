# quantum_transpiler/streamlit_app.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_huggingface import HuggingFaceHub
from quantum_transpiler.main import create_transpiler_graph
from quantum_transpiler.utils import detect_framework
from research_assistant import ResearchAssistant

def app():
    st.title("Quantum Framework Transpiler and Research Assistant")

    # Model Provider Selection
    provider = st.selectbox(
        "Select Model Provider",
        ["OpenAI", "Google AI", "HuggingFace"]
    )

    # API Key Input
    api_key = st.text_input(
        f"Enter {provider} API Key",
        type="password"
    )

    # Model Selection based on provider
    if provider == "OpenAI":
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]
        model = st.selectbox("Select Model", model_options)
        if api_key:
            llm = ChatOpenAI(model=model, api_key=api_key)
    elif provider == "Google AI":
        model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash-8B"]
        model = st.selectbox("Select Model", model_options)
        if api_key:
            llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
    else:  # HuggingFace
        model_options = ["meta-llama/Llama-2-70b-chat-hf", "tiiuae/falcon-180B"]
        model = st.selectbox("Select Model", model_options)
        if api_key:
            llm = HuggingFaceHub(repo_id=model, huggingfacehub_api_token=api_key)

    # Input/Output Code Areas
    input_code = st.text_area("Input Quantum Circuit Code")
    target_framework = st.selectbox(
        "Target Framework",
        ["Qiskit", "TensorFlow Quantum", "PennyLane", "PyTorch Quantum", "CUDA Quantum"]
    )

    if st.button("Translate") and api_key:
        with st.spinner('Translating...'):
            try:
                # Create graph with selected LLM
                graph = create_transpiler_graph(llm)
                source_framework = detect_framework(input_code)
                
                result = graph.invoke({
                    "input_code": input_code,
                    "source_framework": source_framework, 
                    "target_framework": target_framework
                })
                
                st.code(result["translated_code"], language="python")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
    elif st.button("Translate") and not api_key:
        st.warning("Please enter an API key first")

    st.header("Research Assistant")
    topic = st.text_input("Research Topic")
    max_analysts = st.number_input("Max Analysts", min_value=1, max_value=10, value=3)
    human_analyst_feedback = st.text_area("Human Analyst Feedback")

    if st.button("Conduct Research") and api_key:
        with st.spinner('Conducting research...'):
            try:
                assistant = ResearchAssistant()
                result = assistant.conduct_research(
                    topic=topic,
                    max_analysts=max_analysts,
                    human_analyst_feedback=human_analyst_feedback
                )
                st.text_area("Research Report", result["research_report"], height=300)
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
    elif st.button("Conduct Research") and not api_key:
        st.warning("Please enter an API key first")

if __name__ == "__main__":
    app()