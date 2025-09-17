# E.M.A.I.L: Easy Messaging Artificial Intelligence Learner

A personalized, memory-enhanced AI agent designed to automate email triage and draft contextually relevant replies in a user's unique writing style. This project was developed as part of an AI Agent Prototype assignment.

* **Author:** Arpit Ranjan
* **University:** IIT Guwahati
* **Department:** Chemical Science and Technology

---

### Why This Project?

As the branch representative for my department, I receive a high volume of emails from professors and students daily. To manage this communication efficiently and ensure timely, accurate responses, I decided to build this personalized AI assistant. The goal was to create an agent that could not only handle the volume but, more importantly, communicate in my own voice, making the automation feel personal and authentic.

### Agent Architecture

This agent uses a sophisticated, multi-step workflow built with LangGraph. It first triages incoming emails and then, if a response is required, activates a ReAct-style agent to reason, use tools, and draft a reply.

---

## Core Concepts

This project is split into two main parts: fine-tuning a personalized language model and integrating that model into an autonomous agent framework.

### Part 1: The Fine-Tuned Brain (Model Personalization)

This part corresponds to the `model.py` script. The goal here was to create a "digital twin" of my writing style.

#### Why Fine-Tuning? The Importance of Style Adaptation

A generic, pre-trained model like GPT-4 can write a grammatically correct email, but it can't sound like *me*. The core technical challenge of this project was to achieve **stylistic adaptation**. By fine-tuning the `microsoft/Phi-3-mini` model on a curated dataset of my own sent emails, the agent learns my unique patterns:
* **Tone:** It learns whether to be formal ("Respected Sir/Ma'am") or conversational ("Yes ma'am, sounds good.").
* **Phrasing:** It picks up on my common phrases and sentence structures.
* **Context:** It learns the specific ways I interact with different people (e.g., professors vs. administrative staff).

This fine-tuning was performed using the LoRA (Low-Rank Adaptation) technique, which is a highly efficient method for adapting large models without requiring massive computational resources.

### Part 2: The Autonomous Agent (Agentic Workflow)

This part corresponds to the `agent.py` script. Here, the fine-tuned model becomes the "brain" of a larger system that can reason, remember, and use tools.

* **Triage System:** The agent first classifies every email as `ignore`, `notify`, or `respond`. This is handled by a simplified prompt that asks the fine-tuned model for a single-word answer, ensuring high reliability.
* **Response System:** If an email requires a response, a powerful **ReAct (Reasoning and Acting) Agent** is activated. This agent can think step-by-step, decide which tools to use, and process information before drafting a final reply.
* **Memory System:** The agent is connected to a memory store, giving it two key abilities:
    * **Episodic Memory:** It can search for past, user-provided examples to make better triage decisions.
    * **Semantic Memory:** It can search a knowledge base of facts (e.g., "My student ID is...") to add context to its drafted replies.

---

## Getting Started: How to Run This Project

This project is designed to be run in a **Google Colab** environment to leverage its free GPU resources.

### Prerequisites

* A Google Colab account.
* A Hugging Face account (optional, for accessing models).
* Your own `dataset.jsonl` file if you wish to re-train the model on your own style.

### Installation

All required libraries are installed via the Python notebooks. For a manual setup, you can create a `requirements.txt` file and run `pip install -r requirements.txt`.

### Running the Scripts

**Step 1: Fine-Tune Your Personalized Model (`model.py`)**
1.  Upload your training data as `dataset.jsonl` to your Colab session.
2.  Run the fine-tuning script.
3.  This will train the model and save the LoRA adapter to a folder named `./my-email-lora-adapter`.
4.  It's recommended to first run the utility script to save the base `Phi-3-mini` model to your Google Drive for faster loading in the future.

**Step 2: Run the Autonomous Agent (`agent.py`)**
1.  Ensure your fine-tuned adapter (`my-email-lora-adapter`) has been generated and the base model is accessible on Google Drive.
2.  Run the agent script.
3.  The script will load your fine-tuned model, build the agent, and run it on a sample email to demonstrate its capabilities.

---

## Future Work & Scalability

This prototype is a powerful demonstration, but to turn it into a production-ready application, several steps could be taken:

* **Deployment:** The agent would need to be moved from a Colab notebook to a persistent, 24/7 cloud environment (e.g., a Docker container running on a VM in AWS, GCP, or Azure).
* **Real-Time Email Processing:** Instead of manually running the script, it could be connected to the Microsoft Graph API using webhooks. This would allow the agent to process emails in real-time as they arrive.
* **Robust Memory:** The `InMemoryStore` is great for prototyping but is lost when the session ends. For a production system, this would be replaced with a persistent vector database (e.g., Pinecone, Weaviate, or Redis) for the agent's memory.

## Further Applications

The core architecture of this agent (triage -> reason -> act) can be adapted for many other personal automation tasks, such as:
* A meeting summarizer that listens to a conversation and drafts a summary email.
* A personalized news agent that reads articles from various sources and creates a daily digest tailored to your interests.
* An automated calendar assistant that can schedule meetings by communicating with others over email.