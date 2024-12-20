import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Return the transcription
    return result['text']

if __name__ == "__main__":
    file_path = "path_to_your_audio_file.wav"
    transcript = transcribe_audio(file_path)
    print("Transcript:", transcript)