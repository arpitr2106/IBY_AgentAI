 
# PART 1: SETUP (Imports, Libraries, and Keys)

# Import all necessary modules
import torch
import os
import re
from google.colab import drive
from typing import TypedDict, Literal, Annotated, List

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


 
# PART 2: LOAD YOUR FINE-TUNED MODEL

# 2a. Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 2b. Configure Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2c. Load Base Model & Tokenizer from Your Google Drive
model_path = "/content/drive/MyDrive/my_models/phi-3-mini"
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 2d. Load your fine-tuned LoRA adapter
adapter_path = "./my-email-lora-adapter"
tuned_model = PeftModel.from_pretrained(base_model, adapter_path)

# 2e. Create the LangChain LLM object using the NEW library
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model=tuned_model,
    tokenizer=tokenizer
)
model_kwargs = {"eos_token_id": tokenizer.eos_token_id, "max_new_tokens": 250}
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=model_kwargs)

# PART 3: SETUP THE AGENT'S MEMORY
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = InMemoryStore(index={"embed": embedding_model})

# PART 4: DEFINE THE AGENT'S STRUCTURE AND LOGIC

# 4a. Define the Agent's State
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
    triage_result: str

# 4b. Define helper function for examples
def format_few_shot_examples(examples):
    formatted = []
    for eg in examples:
        email, label = eg.value['email'], eg.value['label']
        formatted.append(f"From: {email['author']}\nSubject: {email['subject']}\nBody: {email['email_thread'][:300]}...\n\nClassification: {label}")
    return "\n\n".join(formatted)

# 4c.
def triage_email(state: State, config: dict) -> dict:
    email = state["email_input"]
    user_id = config["configurable"]["langgraph_user_id"]
    
    examples = store.search(("email_assistant", user_id, "examples"), query=str(email))
    formatted_examples = format_few_shot_examples(examples)
    
    # A much simpler prompt asking for just one word
    prompt_str = f"""You are an email triage assistant. Your only job is to classify the following email.
You must respond with only a single word: 'ignore', 'notify', or 'respond'.

Here are some examples of previous classifications:
{formatted_examples}

Now, classify this new email:
From: {email.get("author")}
To: {email.get("to")}
Subject: {email.get("subject")}
Body: {email.get("email_thread")}

Classification:"""

    # Call the LLM
    llm_output = llm.invoke(prompt_str)
    
    # Clean the output to get just the single word
    match = re.search(r'\b(ignore|respond|notify)\b', llm_output.lower())
    classification = match.group(0) if match else "ignore" # Default to 'ignore' if no clear word is found
    
    print(f"Triage Result (Cleaned): {classification}")
    return {"triage_result": classification}

# 4d. Define Tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    print(f"--- TOOL ACTION: SENDING EMAIL ---\nTo: {to}\nSubject: '{subject}'\nContent:\n{content}\n")
    return f"Email sent to {to}."
@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    print(f"--- TOOL ACTION: CHECKING CALENDAR FOR {day} ---")
    return f"Available times on {day}: 9:00 AM, 2:00 PM."
manage_memory_tool = create_manage_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))
search_memory_tool = create_search_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))
tools = [write_email, check_calendar_availability, manage_memory_tool, search_memory_tool]

# 4e. Create the Response Agent with a Transformer

# This is the prompt the ReAct agent will use
prompt = hub.pull("hwchase17/react")

# This is the core agent runnable
agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# This is the Agent Executor engine
agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False, handle_parsing_errors=True)

# This is our transformer function
def format_state_for_agent(state: State) -> dict:
    """Formats the email_input into the 'input' key that the agent expects."""
    email = state['email_input']
    # Create a clear, natural language instruction for the agent
    instruction = (
        f"I have received an email and need to draft a response.\n"
        f"From: {email.get('author')}\n"
        f"Subject: {email.get('subject')}\n"
        f"Body: {email.get('email_thread')}\n\n"
        "Please determine the necessary actions and draft a suitable reply using the available tools."
    )
    return {"input": instruction, "messages": state["messages"]}

# We now chain the transformer and the agent executor together.
response_agent = RunnableLambda(format_state_for_agent) | agent_executor
 
# PART 5: BUILD AND COMPILE THE AGENT GRAPH
workflow = StateGraph(State)
workflow.add_node("triage", lambda state, config: triage_email(state, config))
workflow.add_node("response_agent", response_agent)
def route_based_on_triage(state):
    if state["triage_result"] == "respond":
        return "response_agent"
    else:
        print(f"Triage result is '{state['triage_result']}'. Ending workflow.")
        return END
workflow.add_edge(START, "triage")
workflow.add_conditional_edges("triage", route_based_on_triage, {"response_agent": "response_agent", END: END})
workflow.add_edge("response_agent", END)
checkpointer = MemorySaver()
agent = workflow.compile(checkpointer=checkpointer)

# PART 6: PERSONALIZE THE AGENT'S MEMORY
user_id = "arpit_final_test"
prof_email_example = {"email": {"author": "Sumana", "subject": "Project", "email_thread": "Hi Arpit, can we meet?"}, "label": "respond"}
promo_email_example = {"email": {"author": "Updates", "subject": "New features!", "email_thread": "Check out our new features."}, "label": "ignore"}
store.put(("email_assistant", user_id, "examples"), "prof_example", prof_email_example)
store.put(("email_assistant", user_id, "examples"), "promo_example", promo_email_example)
print("- Added triage examples.")
store.put(("email_assistant", user_id, "collection"), "fact1", "Professor Sumana is my guide for the AI project.")
store.put(("email_assistant", user_id, "collection"), "fact2", "My student ID is 230122010.")
print("- Added semantic facts.")