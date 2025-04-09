from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile

app = Flask(__name__)
CORS(app)

model = whisper.load_model("tiny")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        file.save(temp.name)
        result = model.transcribe(temp.name)
        os.unlink(temp.name)

    return jsonify({'text': result['text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

