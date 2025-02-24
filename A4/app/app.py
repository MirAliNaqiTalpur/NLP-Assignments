import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, BertModel
import torch
# Import your custom similarity functions
from model import calculate_similarity, classify_similarity_label  


app = Flask(__name__)

device = torch.device("cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Load your fine-tuned SBERT model
model.load_state_dict(torch.load("SBERT_finetuned.pth", map_location=device))
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get input data from AJAX request
        premise = request.form['premise']
        hypothesis = request.form['hypothesis']

        # Compute similarity score
        score = calculate_similarity(model, tokenizer, premise, hypothesis, device)
        
        # Convert score to a Python float (this resolves the JSON serialization issue)
        score = float(score)  # Ensure it is a Python native float
        
        # Debug: print the similarity score to check its value
        print("Similarity Score:", score)

        # Get label classification
        label = classify_similarity_label(score)
        
        # Debug: print the label to see what is returned
        print("Label:", label)

        # Return JSON response
        return jsonify({'label': label, 'score': round(score, 4)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
