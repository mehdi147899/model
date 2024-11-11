from flask import Flask, request, jsonify
import MistralModeling  # This will use the notebook's code as a module
import os

app = Flask(__name__)

# Initialize the knowledge graph as in the notebook
G = MistralModeling.nx.DiGraph()

@app.route('/')
def index():
    return "Mistral Modeling Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pdf_path = data.get("pdf_path", "file.pdf")
    text_input = data.get("text_input", "body_of_knowledge_with_embeddings_saved (1).csv")

    # Extract text from PDF if pdf_path is provided, else use text_input
    if pdf_path:
        activity_input = MistralModeling.extract_text_from_pdf(pdf_path)
        if not activity_input:
            activity_input = text_input  # Fallback to text_input if PDF extraction fails
    else:
        activity_input = text_input

    # Further processing can go here, such as calling other model functions
    # Example: prediction = some_processing_function(activity_input)

    response = {"activity_input": activity_input[:100] + "..."}  # Placeholder response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
