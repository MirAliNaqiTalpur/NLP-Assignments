from flask import Flask, render_template, request, jsonify
from utils import load_qa_chain
import json

app = Flask(__name__)

# Load pre-generated answers for the 10 questions
with open("answers.json", "r") as f:
    predefined_answers = json.load(f)

# Load the RAG chain for dynamic queries
qa_chain = load_qa_chain()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    # Ensure the request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    
    # Parse JSON data
    data = request.get_json()
    user_input = data.get("question")
    
    # Validate input
    if not user_input:
        return jsonify({"error": "Question is required"}), 400
    
    # Check if the question is one of the predefined 10
    for qa in predefined_answers:
        if qa["question"].lower() == user_input.lower():
            return jsonify({
                "answer": qa["answer"],
                "source": "Predefined Answer"
            })
    
    # Use RAG for dynamic questions
    try:
        response = qa_chain({"question": user_input})
        return jsonify({
            "answer": response["answer"],
            "source": [doc.metadata["source"] for doc in response["source_documents"]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)