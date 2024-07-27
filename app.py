from flask import Flask, jsonify, request
import json
import requests
from transformers import pipeline

app = Flask(__name__)

# Inicialize o pipeline de resumo
summarizer = pipeline('summarization')

@app.route('/api/news', methods=['GET'])
def get_news():
    # Simule a obtenção de notícias de uma fonte
    with open('politics_output.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/api/summarize', methods=['POST'])
def summarize_news():
    json_data = request.json
    if 'text' not in json_data:
        return jsonify({'error': 'No text provided'}), 400

    text = json_data['text']
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
