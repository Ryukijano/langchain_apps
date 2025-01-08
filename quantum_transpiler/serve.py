# quantum_transpiler/serve.py

from fastapi import FastAPI
from pydantic import BaseModel
from quantum_transpiler.main import create_transpiler_graph
from quantum_transpiler.utils import detect_framework
from research_assistant import ResearchAssistant

app = FastAPI(
    title="Quantum Framework Transpiler API",
    description="API for translating between quantum computing frameworks"
)

class TranspilerRequest(BaseModel):
    source_code: str
    target_framework: str

class ResearchRequest(BaseModel):
    topic: str
    max_analysts: int
    human_analyst_feedback: str

@app.post("/translate")
async def translate_circuit(request: TranspilerRequest):
    graph = create_transpiler_graph()
    source_framework = detect_framework(request.source_code)
    
    result = graph.invoke({
        "input_code": request.source_code,
        "source_framework": source_framework,
        "target_framework": request.target_framework
    })
    
    return {"translated_code": result["translated_code"]}

@app.post("/research")
async def conduct_research(request: ResearchRequest):
    assistant = ResearchAssistant()
    result = assistant.conduct_research(
        topic=request.topic,
        max_analysts=request.max_analysts,
        human_analyst_feedback=request.human_analyst_feedback
    )
    return {"research_report": result}
