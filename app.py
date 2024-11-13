from flask import Flask, request, jsonify, render_template
import MistralModeling  # Assuming this imports your model logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('recommandation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Extract the text and file input from the form data
        input_text = request.form.get("inputText")
        input_image = request.files.get("inputImage")
        
        # Use the correct function from MistralModeling based on the input type
        if input_text:
            # Call the function that handles text input and generates a graph-based response
            result = MistralModeling.generate_response_with_graph(input_text)
        elif input_image:
            # Save the image temporarily and call the function for image-based processing
            image_path = "temp_image.png"
            input_image.save(image_path)
            result = MistralModeling.generate_response_from_image(image_path)  # Correct function for images
        else:
            result = "Please provide an input."
        
        # Return the result as a JSON response
        return jsonify({"recommendation": result})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"recommendation": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
