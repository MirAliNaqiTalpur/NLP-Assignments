from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, BertConfig, AutoModelForSequenceClassification
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# Define the student config (matches training)
student_config = BertConfig(
    num_hidden_layers=6, num_attention_heads=12, hidden_size=768,
    intermediate_size=3072, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    num_labels=3
)
model = AutoModelForSequenceClassification.from_config(student_config)
model.load_state_dict(torch.load("odd_layer_student.pth", map_location=device))
model.to(device)
model.eval()

# Label mapping
id2label = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

# Home route with input form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_text = ""  # Default empty text
    if request.method == 'POST':
        input_text = request.form['text']  # Capture the submitted text
        if input_text:
            # Tokenize input
            inputs = tokenizer(input_text, max_length=128, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                pred_label = id2label[pred_id]
                confidence = probs[0, pred_id].item()

            prediction = f"Classified as: {pred_label} (Confidence: {confidence:.4f})"
    
    # Pass input_text to the template
    return render_template('index.html', prediction=prediction, input_text=input_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)