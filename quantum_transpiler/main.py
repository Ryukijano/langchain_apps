# quantum_transpiler/main.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from .utils import translate_code, validate_translation

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.1
)

QUANTUM_TRANSLATION_PROMPT = """You are a quantum computing expert that translates between different quantum frameworks.
Given quantum code in one framework, translate it to the target framework while maintaining exact functionality.

Rules:
1. Preserve all quantum operations and their order
2. Maintain qubit mapping and connectivity
3. Keep quantum circuit depth equivalent
4. Handle framework-specific optimizations
5. Include required imports

Input code: {input_code}
Source framework: {source_framework}
Target framework: {target_framework}
"""

class TranspilerState(dict):
    """State maintained during translation"""
    input_code: str
    source_framework: str 
    target_framework: str
    translated_code: str = ""
    validation_result: bool = False

def create_transpiler_graph(llm=None):
    """Create translation graph with configurable LLM"""
    if llm is None:
        # Default to Gemini if no LLM provided
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.1
        )

    # Create graph with provided LLM
    state_graph = StateGraph(TranspilerState)
    
    state_graph.add_node("translate", translate_code) 
    state_graph.add_node("validate", validate_translation)

    state_graph.set_entry_point("translate")
    state_graph.add_edge('translate', 'validate')
    state_graph.add_conditional_edges(
        'validate',
        lambda x: 'translate' if not x['valid'] else END
    )

    graph = state_graph.compile()
    return graph