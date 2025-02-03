from my_model import Seq2SeqTransformer, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadAttentionLayer, AdditiveAttention, PositionwiseFeedforwardLayer
import torch
from flask import Flask, request, render_template
import torch.nn.functional as F
import numpy as np
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Initialize Flask app
app = Flask(__name__)

# Load the vocab from the saved file
vocab_transform = torch.load('vocab.pth')

# Access the vocab objects for each language
SRC_VOCAB = vocab_transform['en']
TRG_VOCAB = vocab_transform['sd']

# Access the special token indices directly
SRC_PAD_IDX = SRC_VOCAB['<pad>']
TRG_PAD_IDX = TRG_VOCAB['<pad>']
SOS_IDX = SRC_VOCAB['<sos>']
EOS_IDX = SRC_VOCAB['<eos>']

# Hyperparameters
SRC_VOCAB_SIZE = len(SRC_VOCAB)
TRG_VOCAB_SIZE = len(TRG_VOCAB)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
# attn_variant = "additive"  
attn_variant = "multiplicative" 
enc = Encoder(SRC_VOCAB_SIZE, 256, 3, 8, 512, 0.1, attn_variant, DEVICE)
dec = Decoder(TRG_VOCAB_SIZE, 256, 3, 8, 512, 0.1, attn_variant, DEVICE)

# Initialize model
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)

# Load the saved model checkpoint (the checkpoint is a list)
# checkpoint = torch.load('additive_Seq2SeqTransformer.pt', map_location=DEVICE)
checkpoint = torch.load('multiplicative_Seq2SeqTransformer.pt', map_location=DEVICE)


# Check if the checkpoint is a list and handle it accordingly
if isinstance(checkpoint, list):
    state_dict = checkpoint[0]  # Assuming the first element in the list is the state_dict
else:
    state_dict = checkpoint  # If it's already a dictionary, use it directly

# Load the state_dict into the model
model.load_state_dict(state_dict, strict=False)


# Function to translate input sentence
def translate_sentence(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    src_vocab = vocab_transform['en']
    trg_vocab = vocab_transform['sd']
    src_stoi = src_vocab.get_stoi()
    trg_stoi = trg_vocab.get_stoi()
    trg_itos = trg_vocab.get_itos() 

    src_tokens = [src_stoi[word] if word in src_stoi else src_stoi['<unk>'] for word in sentence.split()]
    src = torch.tensor(src_tokens).unsqueeze(0).to(device)  

    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
        trg = torch.tensor([[trg_stoi['<sos>']]], device=device)
        trg_mask = model.make_trg_mask(trg)

        max_translation_length = min(len(src_tokens) * 2, max_len)  # Max 2x words

        for _ in range(max_translation_length):
            output, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()

            if pred_token == trg_stoi['<eos>']:
                break

            new_token = torch.tensor([[pred_token]], dtype=torch.long, device=device)
            trg = torch.cat((trg, new_token), dim=1)
            trg_mask = model.make_trg_mask(trg)

    trg_tokens = trg.squeeze(0).cpu().numpy()
    trg_sentence = ' '.join([trg_itos[i] for i in trg_tokens if i not in {trg_stoi["<sos>"], trg_stoi["<eos>"]}])

    return trg_sentence

# Flask route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle translation
@app.route('/translate', methods=['POST'])
def translate():
    sentence = request.form['sentence']
    translated_sentence = translate_sentence(sentence, model, vocab_transform, DEVICE)
    return render_template('index.html', original=sentence, translated=translated_sentence)

if __name__ == '__main__':
    app.run(debug=True)
