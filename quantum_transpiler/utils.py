# quantum_transpiler/utils.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def detect_framework(code: str) -> str:
    """Detect source framework from code"""
    framework_signatures = {
        "qiskit": ["QuantumCircuit", "qiskit"],
        "tfq": ["tfq", "cirq", "tensorflow_quantum"],
        "pennylane": ["qml", "pennylane"],
        "torch_quantum": ["tq", "torch_quantum"], 
        "cuda_quantum": ["cudaq", "cuda_quantum"]
    }
    
    for framework, signatures in framework_signatures.items():
        if any(sig in code for sig in signatures):
            return framework
    return "unknown"

def validate_translation(state: dict) -> bool:
    """Validate translated code"""
    if not state.get("translated_code"):
        return {"validation_result": False}
    return {"validation_result": True}

def get_translation_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate this quantum code to the target framework."),
        ("human", "{input_code}")
    ])
    
    return prompt | model

def translate_code(state: dict):
    """Translate quantum code between frameworks"""
    translation_chain = get_translation_chain()
    result = translation_chain.invoke({
        "input_code": state["input_code"],
        "source_framework": state["source_framework"],
        "target_framework": state["target_framework"]
    })
    return {"translated_code": result.content}