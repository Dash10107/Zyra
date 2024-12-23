import os 
from transformers import pipeline
from whisper import Whisper
whispermodel = whisper.load_model("base")  # Choose your Whisper model size
def translate(audio_file):
    upload_folder = os.path.abspath("uploads")
    audio_path = os.path.join(upload_folder, audio_file.filename)

    # Save the uploaded audio file
    audio_file.save(audio_path)

    # Transcribe audio using Whisper
    transcription_result = whispermodel.transcribe(audio_path)
    transcription_text = transcription_result['text']
    
    # Use Hugging Face Transformers for summarization
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # Lightweight summarization model
    summary = summarizer(transcription_text, max_length=100, min_length=30, do_sample=False)
    # Cleanup the uploaded file
    os.remove(audio_path)

    # Return the summarized text
    text =  summary[0]['summary_text']
    return text