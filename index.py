from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='sys.stdout', level=logging.INFO, 
                    format='%(asctime)s - %(message)s\n\n', datefmt='%Y-%m-%d %H:%M:%S')

# Load the model
model1 = BertForTokenClassification.from_pretrained("./Model", from_tf=False, from_flax=False)
tokenizer1=BertTokenizerFast.from_pretrained("./Model")
ner_pipeline = pipeline("ner", model=model1, tokenizer=tokenizer1, device=0 if torch.cuda.is_available() else -1)

def predict_ner_tags(text):
    ner_results = ner_pipeline(text)
    labels = []
    current_token = ""
    current_label = None
    start_pos = None

    for result in ner_results:
        token = result["word"]
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                labels.append({
                    "word": current_token,
                    "entity": current_label,
                    "start": start_pos,
                    "end": result["start"]
                })
            current_token = token
            current_label = result["entity"]
            start_pos = result["start"]
    
   
    if current_token:
        labels.append({
            "word": current_token,
            "entity": current_label,
            "start": start_pos,
            "end": ner_results[-1]["end"]
        })

    return labels

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['text']
    predicted_ner_tags = predict_ner_tags(input_text)
    logging.info(f'Input: {input_text}\nPrediction: {predicted_ner_tags}')
    return jsonify(predicted_ner_tags)




if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:8081")
    app.run(debug=True, host='127.0.0.1', port=8081)
