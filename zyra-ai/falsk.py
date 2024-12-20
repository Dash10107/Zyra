from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import whisper
import json

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Directory for temporary files
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    # Check if a file is included in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    # Save the uploaded audio file
    audio_file = request.files['audio']
    file_id = str(uuid.uuid4())  # Generate a unique ID for the file
    audio_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
    audio_file.save(audio_path)

    try:
        # Call the transcription function
        transcription = transcribe_audio(audio_path)
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

def transcribe_audio(audio_path):

    """
    return the transcript of the audio file at the given path
        """
    # Load the Whisper model
    model = whisper.load_model("base")
    # Load the audio file
    audio = whisper.Audio.from_file
    # Transcribe the audio file
    transcription = model.transcribe(audio)
    # Return the transcription
    return transcription['text']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
