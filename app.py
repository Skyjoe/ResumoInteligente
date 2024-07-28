from flask import Flask, jsonify, request
import json
from gradio_client import Client
import os

app = Flask(__name__)

# Acessar o token da variável de ambiente
hf_api_token = os.getenv('HF_API_TOKEN')

# Inicialize o cliente do Gradio
client = Client("Skyjoe/OpenCHAT-mini", api_key=hf_api_token)

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
    
    # Usar o cliente do Gradio para gerar o resumo
    result = client.predict(
        message={"text": text, "files": []},
        api_name="/chat"
    )
    
    summarized_text = result['text']
    
    return jsonify({'summary': summarized_text})

if __name__ == '__main__':
    app.run(debug=True)


