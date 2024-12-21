import os
from transformers import pipeline
import whisper

# Load Whisper model
model = whisper.load_model("base")  # Choose your Whisper model size

# how to import this function in the routes.py file

def summarize_recordings(audio_file):
    # Set up upload directory
    upload_folder = os.path.abspath("uploads")
    os.makedirs(upload_folder, exist_ok=True)  # Ensure the folder exists
    audio_path = os.path.join(upload_folder, audio_file.filename)

    # Save the uploaded audio file
    audio_file.save(audio_path)

    # Transcribe audio using Whisper
    transcription_result = model.transcribe(audio_path)
    transcription_text = transcription_result['text']

    # Use Hugging Face Transformers for summarization
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # Lightweight summarization model
    summary = summarizer(transcription_text, max_length=100, min_length=30, do_sample=False)

    # Cleanup the uploaded file
    os.remove(audio_path)

    # Return the summarized text
    return summary[0]['summary_text']


