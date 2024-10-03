import os
import base64
import tempfile
import atexit
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes

# Step 1: Decode the Base64 encoded JSON key from environment variable
encoded_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if not encoded_key:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set.")

# Decode the Base64 string
decoded_key = base64.b64decode(encoded_key).decode('utf-8')

# Write the JSON key to a temporary file
with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_key_file:
    temp_key_file.write(decoded_key)
    temp_key_file_path = temp_key_file.name

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to the temp file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_file_path

# Ensure the temporary file is deleted when the program exits
def cleanup():
    try:
        os.remove(temp_key_file_path)
    except Exception as e:
        print(f"Error deleting temp key file: {e}")

atexit.register(cleanup)

# Step 2: Create Prompt Template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Step 3: Initialize the Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  # Use the desired Gemini model
    temperature=0.7  # Set desired temperature
)

# Step 4: Initialize the Output Parser
parser = StrOutputParser()

# Step 5: Chain the components using LCEL
chain = prompt_template | model | parser

# Step 6: Define the FastAPI app
app = FastAPI(
    title="LangChain Translation API",
    version="1.0",
    description="An API server for translating text using LangChain and Google's Gemini models.",
)

# Optional: Add a root route for testing connectivity
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Step 7: Add the chain as a route
add_routes(
    app,
    chain,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
