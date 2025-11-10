import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import traceable
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.runnables import RunnableSequence

# Load environment variables
load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "LangSmith-Local-Demo")

if not LANGSMITH_API_KEY:
    print("❌ Set LANGSMITH_API_KEY in your .env file to use this demo")
    exit(1)

# Initialize LangSmith client
client = Client(api_key=LANGSMITH_API_KEY)
print("✅ LangSmith client initialized")

# Ensure project exists (or create)
try:
    project = client.read_project(project_name=LANGCHAIN_PROJECT)
    print(f"📂 Found existing project: {project.name}")
except Exception:
    print(f"🆕 Creating new LangSmith project: {LANGCHAIN_PROJECT}")
    project = client.create_project(project_name=LANGCHAIN_PROJECT)

# --- Local Hugging Face model (CPU) ---
generator = pipeline("text-generation", model="distilgpt2", max_new_tokens=100, device=-1)
local_llm = HuggingFacePipeline(pipeline=generator)

# --- Modern RunnableSequence chain ---
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short explanation about {topic} in simple terms."
)
chain = RunnableSequence(first=prompt, last=local_llm)

# We’ll store the run_id from LangSmith here
last_run_id = None


@traceable(name="LangSmith Local LLM Demo")
def run_llm_chain(topic: str):
    """Run LLM chain with LangSmith tracing enabled."""
    from langsmith.run_helpers import get_current_run_tree

    result = chain.invoke({"topic": topic})
    # Capture the current run_id for feedback logging
    global last_run_id
    run_tree = get_current_run_tree()
    if run_tree:
        last_run_id = run_tree.id
    return result


if __name__ == "__main__":
    topic = "Retrieval-Augmented Generation (RAG)"
    print(f"🚀 Running local LLM demo on topic: {topic}\n")

    result = run_llm_chain(topic)
    print("🔹 Model Output:")
    print(result)

    # ✅ Attach feedback to the captured run_id
    if last_run_id:
        feedback = client.create_feedback(
            run_id=last_run_id,
            key="clarity",
            score=0.9,
            comment="Output successfully generated and logged to LangSmith."
        )
        print(f"\n✅ Feedback submitted for run_id: {last_run_id}")
    else:
        print("\n⚠️ Could not find run_id to attach feedback (check tracing config).")

    print(f"🔗 Project: {LANGCHAIN_PROJECT}")
    print("🔗 View your run at: https://smith.langchain.com")
