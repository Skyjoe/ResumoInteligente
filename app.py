from flask import Flask, jsonify, request
import json
from transformers import pipeline
import os

app = Flask(__name__)

# Acessar o token da variável de ambiente
hf_api_token = os.getenv('HF_API_TOKEN')

# Inicialize o pipeline de resumo com o modelo ChatGPT mini
summarizer = pipeline(
    "text-generation",  # Tipo de tarefa pode ser "text-generation"
    model="Skyjoe/OpenCHAT-mini",  # Substitua pelo identificador correto do seu modelo
    tokenizer="Skyjoe/OpenCHAT-mini",  # O mesmo modelo geralmente é usado como tokenizer
    api_key=hf_api_token
)

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
    
    # Usar o summarizer para gerar o resumo
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    summarized_text = summary[0]['generated_text'] if 'generated_text' in summary[0] else summary[0]['summary_text']
    
    return jsonify({'summary': summarized_text})

if __name__ == '__main__':
    app.run(debug=True)

