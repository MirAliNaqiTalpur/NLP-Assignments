from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the pre-trained models (you need to pickle these models)
with open('embedding/glove_embeds.pickle', 'rb') as f:
    glove_model = pickle.load(f)

with open('embedding/neg_embeds.pickle', 'rb') as f:
    neg_model = pickle.load(f)

with open('embedding/skipgram_embeds.pickle', 'rb') as f:
    skipgram_model = pickle.load(f)

app = Flask(__name__)

def dot_product(query, model):
    # Check if the query exists in the model's vocabulary
    if query not in model:
        print(f"Warning: '{query}' not found in vocabulary. Using default embedding.")
        query = '<UNK>'  # Use a placeholder for unknown words

    query_embedding = model[query]

    # Instead of accessing "corpus", just iterate through the model's word embeddings
    similarities = {}
    for word, embedding in model.items():  # Assuming model is a dictionary of word embeddings
        similarity = np.dot(query_embedding, embedding)  # Compute dot product
        similarities[word] = similarity
    
    # Sort the similarities in descending order and return the top 10 most similar words
    top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_similar


@app.route("/", methods=["GET", "POST"])
def home():
    top_similar_words = []
    if request.method == "POST":
        query = request.form["query"]
        model_choice = request.form["model_choice"]
        
        if model_choice == "glove":
            top_similar_words = dot_product(query, glove_model)
        elif model_choice == "neg":
            top_similar_words = dot_product(query, neg_model)
        elif model_choice == "skipgram":
            top_similar_words = dot_product(query, skipgram_model)

    return render_template("index.html", top_similar_words=top_similar_words)

if __name__ == "__main__":
    app.run(debug=True)
