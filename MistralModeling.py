#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os 
import llama_cpp
import pandas as pd


# In[24]:


import networkx as nx
import pandas as pd

# Initialize the knowledge graph as a directed graph
G = nx.DiGraph()

# Debug message
print("Knowledge graph initialized.")


# In[25]:


import fitz  # PyMuPDF for PDF text extraction

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text += page.get_text("text")
        print("Successfully extracted text from PDF.")
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        text = ""
    return text


# In[26]:


# Input cell for activity source
activity_input = None  # Placeholder for activity text
pdf_path = "file.pdf"  # Set PDF path here if using a PDF, or leave as an empty string

# Full task description as text input for dynamic activity setting
text_input = (
    "Provide a detailed 7-day mobile app development plan. Outline each day’s tasks, key milestones, and expected deliverables. "
    "Include a brief risk assessment with potential risks and mitigation strategies. Keep responses concise and brief."
)

# Conditional logic to determine whether to use PDF or text input
use_pdf = pdf_path != ""

# Set activity_input based on whether to use PDF or text input
if use_pdf:
    activity_input = extract_text_from_pdf(pdf_path)
    if not activity_input:
        print("Warning: PDF text extraction failed, falling back to text input.")
        activity_input = text_input
else:
    activity_input = text_input

print("Activity input set successfully:", activity_input[:100] + "..." if len(activity_input) > 100 else activity_input)


# In[27]:


# Set activity_input based on whether to use PDF or text input
if use_pdf:
    activity_input = extract_text_from_pdf(pdf_path)
    if not activity_input:
        print("Warning: PDF text extraction failed, falling back to text input.")
        activity_input = text_input
else:
    activity_input = text_input

print("Activity input set successfully:", activity_input[:100] + "..." if len(activity_input) > 100 else activity_input)


# # Gnerating Graph

# In[29]:


import pandas as pd
import networkx as nx  # Assuming you're using NetworkX for graph handling

# Load the dataset into a DataFrame
df = pd.read_csv("body_of_knowledge_with_embeddings_saved (1).csv")

# Initialize the graph
G = nx.DiGraph()  # You can choose another graph type based on your needs

# Define parse_inputs function if not yet defined, for example:
def parse_inputs(inputs_str):
    # Replace with actual parsing logic; here it's assuming inputs are comma-separated
    return [item.strip() for item in inputs_str.split(",")]

# Populate the graph with nodes and edges based on the dataset
for idx, row in df.iterrows():
    # Add process as nodes with a default name if missing
    process_id = row["Process_ID"]
    process_name = row.get("Process_Name", f"Unnamed_Process_{process_id}")
    knowledge_area = row["Knowledge_Area_Name"]
    process_description = row["Process_Description"]
    
    # Node attributes include process details for reasoning
    G.add_node(process_id, name=process_name, area=knowledge_area, description=process_description)
    
    # Manually parse 'Inputs' for dependencies, if 'Inputs' is a string
    inputs = []
    if isinstance(row["Inputs"], str):
        try:
            inputs = parse_inputs(row["Inputs"])
        except Exception as e:
            print(f"Error parsing Inputs for row {idx}: {e}")
            inputs = []

    # Ensure each parsed input is hashable and add edges
    for input_item in inputs:
        if isinstance(input_item, (str, int)):  # Check if input is hashable
            G.add_edge(input_item, process_id, relationship="dependency")
        else:
            print(f"Skipped non-hashable input for process '{process_name}': {input_item}")

    # Debug message for node and edge addition
    print(f"Added process '{process_name}' with dependencies: {inputs if inputs else 'None'}")

print("Knowledge graph populated with processes and dependencies.")


# # Reasoing logic for LLama

# In[30]:


# Define function to get dependencies of a given process
def get_dependencies(process_id):
    dependencies = list(G.predecessors(process_id))
    print(f"Dependencies for '{G.nodes[process_id]['name']}': {dependencies}")
    return dependencies

# Define function to get stakeholders for a process (based on edge relationships)
def get_stakeholders(process_id):
    stakeholders = [node for node, attr in G.nodes(data=True) if attr.get("area") == "Stakeholder Management"]
    print(f"Stakeholders for '{G.nodes[process_id]['name']}': {stakeholders}")
    return stakeholders

# Define function to identify risk-related dependencies (mocked for simplicity)
def get_risks(process_id):
    risks = [node for node, attr in G.nodes(data=True) if attr.get("area") == "Risk Management"]
    print(f"Risks affecting '{G.nodes[process_id]['name']}': {risks}")
    return risks

# Debugging statements to verify function outputs
print("Reasoning functions initialized.")


# In[31]:


# Define a function for Llama to query the graph
def generate_graph_context(activity_input):
    # Find the process ID associated with the activity input
    process_ids = [pid for pid, data in G.nodes(data=True) if data['name'].lower() in activity_input.lower()]
    if not process_ids:
        print("No matching process found in the knowledge graph.")
        return "No relevant process found in the knowledge graph."

    process_id = process_ids[0]  # Assuming the first match
    dependencies = get_dependencies(process_id)
    stakeholders = get_stakeholders(process_id)
    risks = get_risks(process_id)

    # Construct context for Llama's input prompt
    context = (
        f"Activity: {activity_input}\n"
        f"Dependencies: {dependencies}\n"
        f"Stakeholders: {stakeholders}\n"
        f"Risks: {risks}\n"
    )

    # Debug message for constructed context
    print(f"Constructed graph context for Llama:\n{context}")
    return context


# # Initializing data and model 

# In[32]:


import os
import pandas as pd
import llama_cpp  # Ensure llama_cpp is installed and configured

# Specify paths for the model and dataset
model_path = os.path.join("Llama-3.2-1B-Instruct-Q5_K_M.gguf")
dataset_path = os.path.join("pmbok_prompt_completion_pairs (1).csv")

# Initialize the llama-cpp model with the specified model path
llama = llama_cpp.Llama(model_path=model_path, verbose=True)

# Load the dataset
dataset = pd.read_csv(dataset_path)

# Verify the data and model loading
print("Model loaded successfully.")
print("Dataset loaded successfully. Sample data:")
print(dataset.head())


# # Formatting

# In[20]:


def prepare_prompt_completion(example):
    prompt = example["Prompt"]
    completion = example["Completion"]
    return f"{prompt}\n{completion}"

# Convert the dataset into prompt-completion pairs
formatted_prompts = [prepare_prompt_completion(row) for _, row in dataset.iterrows()]


# # Fine tuning Llama-3.2-1B Qunatized version. 

# In[21]:


# Step 2: Define Training Loop for Fine-Tuning
for step, text in enumerate(formatted_prompts):
    
    print(f"Training Step {step+1}/{len(formatted_prompts)}")
    
    
    output = llama(text)
    print("Output:", output["choices"][0]["text"])
    
    

print("finetuning complete.")


# # Activity input and output generation 

# In[23]:


def generate_graph_context(activity_input):
    # Lowercase input for case-insensitive matching
    input_keywords = set(activity_input.lower().split())

    # Try to find process IDs by matching keywords in `name` or `description`
    matching_process_ids = []
    for pid, data in G.nodes(data=True):
        name_keywords = set(data.get("name", "").lower().split())
        description_keywords = set(data.get("description", "").lower().split())
        
        # Check if there’s any overlap between input keywords and node keywords
        if input_keywords & name_keywords or input_keywords & description_keywords:
            matching_process_ids.append(pid)
    
    if not matching_process_ids:
        print("No exact match found for the input. Attempting fallback context.")
        return "No relevant process found in the knowledge graph."

    process_id = matching_process_ids[0]
    
    # Filter out any numerical-only dependencies, stakeholders, or risks
    dependencies = [dep for dep in get_dependencies(process_id) if isinstance(dep, str)]
    stakeholders = [stake for stake in get_stakeholders(process_id) if isinstance(stake, str)]
    risks = [risk for risk in get_risks(process_id) if isinstance(risk, str)]

    # Construct context for Llama's input prompt
    context = (
        f"Activity: {activity_input}\n"
        f"Dependencies: {dependencies}\n"
        f"Stakeholders: {stakeholders}\n"
        f"Risks: {risks}\n"
    )

    # Debug message for constructed context
    print(f"Constructed graph context for Llama:\n{context}")
    return context


# In[ ]:


def generate_response_with_graph(activity_input):
    # Retrieve context from the graph
    graph_context = generate_graph_context(activity_input)

    # Combine graph context with the activity description from `activity_input`
    prompt = (
        f"{graph_context}\n\n"
        f"{activity_input} Include a brief risk assessment with potential risks and mitigation strategies. "
        "Keep responses concise and brief."
    )
    
    # Print the final prompt sent to Llama for debugging
    print("Final Prompt to Llama:\n", prompt)

    # Generate response using Llama with the new prompt
    response = llama(prompt, max_tokens=5500)

    # Extract the text from the response and print it
    output_text = response["choices"][0]["text"]
    print("Model Response:", output_text)
    return output_text

# Run the updated function with dynamic activity_input
Fine_tuned_Response = generate_response_with_graph(activity_input)


# # Extract Relevant Information from Llama’s Output

# In[32]:


import re

def extract_entities_from_output(output_text):
    """
    Extract tasks, dependencies, risks, and stakeholders from Llama's output.
    This function uses regex to identify key phrases for simplicity.
    """
    tasks = re.findall(r"Tasks?: (.+?)(?:\n|$)", output_text)
    dependencies = re.findall(r"Dependencies?: (.+?)(?:\n|$)", output_text)
    risks = re.findall(r"Risks?: (.+?)(?:\n|$)", output_text)
    stakeholders = re.findall(r"Stakeholders?: (.+?)(?:\n|$)", output_text)

    # Debug output for extracted entities
    print("Extracted Tasks:", tasks)
    print("Extracted Dependencies:", dependencies)
    print("Extracted Risks:", risks)
    print("Extracted Stakeholders:", stakeholders)
    
    return tasks, dependencies, risks, stakeholders


# # Apply Ponderation Techniques

# In[37]:


# Define keyword sets for relevance scoring
task_keywords = {"task", "milestone", "deliverable", "complete"}
dependency_keywords = {"dependency", "requirement", "before", "after"}
risk_keywords = {"risk", "issue", "challenge", "mitigation"}
stakeholder_keywords = {"stakeholder", "team", "manager", "client", "involved"}

def score_entity(entity, keywords):
    """
    Assign a relevance score based on the presence of keywords.
    Higher scores for entities with multiple relevant keywords.
    """
    words = set(entity.lower().split())
    score = sum(1 for word in words if word in keywords)
    return score


# In[38]:


def ponderate_entities(entities, keywords):
    """
    Filter and rank entities based on contextual relevance using keyword scoring.
    """
    scored_entities = [(entity, score_entity(entity, keywords)) for entity in entities]
    # Filter out entities with a score of 0 (no relevance)
    scored_entities = [(entity, score) for entity, score in scored_entities if score > 0]
    # Sort by score in descending order
    ranked_entities = sorted(scored_entities, key=lambda x: x[1], reverse=True)
    
    # Debug output for scored entities
    print("Scored and Ranked Entities:", ranked_entities)
    
    # Return only the entity names, not the scores, for final output
    return [entity for entity, score in ranked_entities]


# 
# # Graph expansion if applicable

# In[39]:


def expand_graph_with_llama_output(output_text):
    # Extract entities from Llama’s output
    tasks, dependencies, risks, stakeholders = extract_entities_from_output(output_text)
    
    # Ponderate each category with contextual scoring
    tasks = ponderate_entities(tasks, task_keywords)
    dependencies = ponderate_entities(dependencies, dependency_keywords)
    risks = ponderate_entities(risks, risk_keywords)
    stakeholders = ponderate_entities(stakeholders, stakeholder_keywords)
    
    # Add tasks as new nodes
    for task in tasks:
        G.add_node(task, type="task")
        print(f"Added task node: {task}")
    
    # Add dependencies as edges between tasks if applicable
    for dependency in dependencies:
        task_links = dependency.split(" and ")
        if len(task_links) == 2:
            G.add_edge(task_links[0].strip(), task_links[1].strip(), relationship="dependency")
            print(f"Added dependency edge: {task_links[0].strip()} -> {task_links[1].strip()}")
    
    # Add risks and link them to tasks
    for risk in risks:
        G.add_node(risk, type="risk")
        for task in tasks:
            G.add_edge(risk, task, relationship="risk_impact")
            print(f"Added risk impact edge: {risk} -> {task}")
    
    # Add stakeholders and link them to tasks
    for stakeholder in stakeholders:
        G.add_node(stakeholder, type="stakeholder")
        for task in tasks:
            G.add_edge(stakeholder, task, relationship="involvement")
            print(f"Added stakeholder involvement edge: {stakeholder} -> {task}")


# In[40]:


# Example: Generate a response and expand the graph with new information
activity_input = "Identify additional tasks and risks for the mobile app project lifecycle"
response_text = generate_response_with_graph(activity_input)  # Assuming this generates a response

# Expand the graph using Llama's output
expand_graph_with_llama_output(response_text)


# # Refinig output with a more perfomrant models

# In[31]:


import os
from groq import Groq

# Loading API
with open("pmkey.txt", "r") as file:
    api_key = file.read().strip()
os.environ["GROQ_API_KEY"] = api_key

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Define a function to refine the already-generated model output using Groq API
def refine_with_groq(model_output):
    messages = [
        {
            "role": "user",
            "content": f"Refine the following project plan to be more grounded in reality and well-structured:\n\n{model_output}"
        }
    ]
    
    # Send request to Groq API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192" 
    )
    
    # Retrieve and print the refined output
    refined_output = chat_completion.choices[0].message.content
    print("Refined Output from Groq API:", refined_output)
    return refined_output

# Call refine_with_groq to refine this stored response
refined_output = refine_with_groq(Fine_tuned_Response)


# # Human evaluation and other Metrics

# In[9]:


from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

reference_text = """
Day 1: Define project scope, objectives, and team roles. 
Day 2: Requirements gathering and stakeholder analysis...
"""  

# ROUGE Score
rouge = Rouge()
rouge_scores = rouge.get_scores(refined_output, reference_text)
print("ROUGE Scores:", rouge_scores)

# Cosine Similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
output_embedding = model.encode(refined_output, convert_to_tensor=True)
reference_embedding = model.encode(reference_text, convert_to_tensor=True)
cosine_similarity = util.pytorch_cos_sim(output_embedding, reference_embedding)
print("Cosine Similarity:", cosine_similarity.item())

