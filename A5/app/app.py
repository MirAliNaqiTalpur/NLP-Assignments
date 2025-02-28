# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer from Hugging Face Hub
model_name = "mirali111/dpo-gpt2-model"  # Update with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get input text from request
        data = request.json
        def format_prompt(input_text):
        # Format similar to how the Anthropic dataset formats prompts
        # This is a simplified example - check actual dataset format
            return f"Human: {input_text}\n\nAssistant:"
        
        # Then in your generate route:
        input_text = format_prompt(data.get('input_text', ''))
        
        if not input_text:
            return jsonify({"error": "Please provide input text"}), 400
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.5,
                top_p=0.92,
                 repetition_penalty=1.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Get just the newly generated text (remove input prompt)
        response_text = generated_text[len(input_text):] if generated_text.startswith(input_text) else generated_text
        
        return jsonify({
            "input": input_text,
            "response": response_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)