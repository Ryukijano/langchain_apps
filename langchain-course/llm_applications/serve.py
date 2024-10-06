import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_core.tools import tool, Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# Step 1: Set up OpenAI (GitHub) credentials
if not os.getenv("GITHUB_TOKEN"):
    raise ValueError("GITHUB_TOKEN is not set")

os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")
os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com/"

# Step 2: Create Prompt Template for Reasoning
system_template = "You are an AI assistant skilled in reasoning and problem-solving. Answer the following question:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{question}')
])

# Step 3: Initialize the OpenAI GPT-4 Model
model = ChatOpenAI(
    model="gpt-4o",  # Use GPT-4 model
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7  # Set desired temperature
)

# Step 4: Initialize the Output Parser
parser = StrOutputParser()

# Step 5: Define Tools
@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

tools = [
    Tool(name="add", func=add, description="Add two numbers"),
    Tool(name="subtract", func=subtract, description="Subtract two numbers"),
    Tool(name="multiply", func=multiply, description="Multiply two numbers"),
    Tool(name="divide", func=divide, description="Divide two numbers")
]

# Step 6: Integrate Memory
memory = MemorySaver()  # Initialize MemorySaver

# Step 7: Create StateGraph and ToolNodes
state_graph = StateGraph(MessagesState)

# Define nodes: these do the work
def assistant(state: MessagesState):
    return {"messages": [model.invoke([SystemMessage(content=system_template)] + state["messages"])]}

# Add nodes to the StateGraph
state_graph.add_node("assistant", assistant)
state_graph.add_node("tools", ToolNode(tools))

# Add edges to the StateGraph
state_graph.add_edge(START, "assistant")
state_graph.add_conditional_edges("assistant", tools_condition)
state_graph.add_edge("tools", END)

# Step 8: Compile the Graph with MemorySaver
react_graph_memory = state_graph.compile(checkpointer=memory)

# Step 9: Create Agent
agent = AgentExecutor(
    model=model,
    memory=memory,
    tools=tools,
    prompt_template=prompt_template,
    output_parser=parser
)

# Step 10: Define the FastAPI app
app = FastAPI(
    title="LangChain Reasoning API",
    version="1.0",
    description="An API server for reasoning and problem-solving using LangChain and OpenAI's GPT-4o model.",
)

# Optional: Add a root route for testing connectivity
@app.get("/")
def read_root():
    return {"message": "Welcome to the LangChain Reasoning API"}

# Add other routes as needed
add_routes(app, agent)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
