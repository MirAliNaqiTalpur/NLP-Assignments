import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
import pickle

# Define the model architecture (replace with your actual architecture)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, 512).to(device),
                torch.zeros(2, batch_size, 512).to(device))
        
# Tokenizer function
def simple_tokenize(sentences):
    """
    Tokenizes a list of sentences into words.
    This is a simple example and may need adjustments based on your data.
    """
    return [sentence.lower().split() for sentence in sentences]

# Load model and vocab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'lstm_model.pth'
vocab_path = 'vocab.pkl'

model = LSTMModel(vocab_size=31866, emb_dim=512, hid_dim=512, num_layers=2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

itos_vocab = {idx: token for token, idx in vocab.items()}

# Initialize Flask
app = Flask(__name__)

def generate_text(prompt, max_seq_len, temperature):
    model.eval()

    # Tokenize prompt
    tokens = simple_tokenize([prompt])[0]
    indices = [vocab.get(t, vocab['<unk>']) for t in tokens]

    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)

            # Softmax and temperature scaling
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            # Handle <unk> and <eos>
            while prediction == vocab.get('<unk>', -1):
                prediction = torch.multinomial(probs, num_samples=1).item()
            if prediction == vocab.get('<eos>', -1):
                break

            indices.append(prediction)

    tokens = [itos_vocab.get(i, '<unk>') for i in indices]
    return ' '.join(tokens)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        temperature = float(request.form['temperature'])
        max_seq_len = int(request.form['max_seq_len'])
        generated_text = generate_text(prompt, max_seq_len, temperature)
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    return render_template('index.html', prompt='', generated_text='')

if __name__ == '__main__':
    app.run(debug=True)