from flask import Flask, request, jsonify, send_file
import json
import os
import tensorflow as tf
from keras.models import load_model
from transformers import TFBertModel
from transformers import BertTokenizer


app = Flask(__name__)

model = tf.keras.models.load_model('opinionfactmodel.h5',custom_objects={'TFBertModel':TFBertModel})

def prepareData(inputText, tokenizer):
    token = tokenizer.encode_plus (
        inputText,
        max_length = 256,
        truncation = True,
        padding = 'max_length',
        add_special_tokens = True,
        return_tensors = 'tf'

    )
    return {
        'inputIds': tf.cast(token.input_ids, tf.float64),
        'attentionMask': tf.cast(token.attention_mask, tf.float64)
    }

@app.route('/')
def home():
    return "Aletheia Classify Server (Flask)"

@app.route('/classify', methods=["POST"])
def classify():
    opinion = []
    fact = []
    data = request.json
    sentences = data.get("sentences")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    for sentence in sentences:
        tokenizedTextInput = prepareData(sentence, tokenizer)
        probs = model.predict(tokenizedTextInput)
        opinion.append(float(probs[0][0]))
        fact.append(float(probs[0][1]))

    return jsonify({'opinion' : opinion, 'fact': fact})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
