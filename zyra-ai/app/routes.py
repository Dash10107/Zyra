from flask import Blueprint, request, jsonify
import whisper
from googletrans import Translator
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import google.generativeai as genai
# Load Whisper model
model = whisper.load_model("base")  # Choose your Whisper model size
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
qa_tokenizer = AutoTokenizer.from_pretrained("t5-small")


bp = Blueprint('api', __name__)

translator = Translator()

@bp.route('/process-audio', methods=['POST'])
def process_audio():
    # Check if audio file is included in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    # <FileStorage: 'audio_gRqgwaCsLRvYYX9CAAAF.wav' ('audio/wave')>

    upload_folder = os.path.abspath("uploads")
    translations_folder = os.path.abspath("transcriptions")
    os.makedirs(translations_folder, exist_ok=True)
    audio_path = os.path.join(upload_folder, audio_file.filename)

    audio_file.save(audio_path)

    # Run Whisper to get transcription
    transcription_result =   model.transcribe(audio_path)
    transcription_text = transcription_result['text']

    # Translate transcription to preferred language (example: Spanish)
    target_language = request.form.get('language', 'es')  # Default to Spanish
    translated_text = translator.translate(transcription_text, dest=target_language).text
    transcription_file_path = os.path.join(translations_folder, f"{os.path.splitext(audio_file.filename)[0]}_transcription.txt")
    with open(transcription_file_path, "w", encoding="utf-8") as translation_file:
        translation_file.write(transcription_text)

    # # Cleanup the uploaded file
    os.remove(audio_path)

    return jsonify({
        "transcription": transcription_text,
        "translation": translated_text
    })

@bp.route("/summarize-recordings", methods=['POST'])
async def summarize_recordings():
    # Check if audio file is included in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    # Set up upload directory
    upload_folder = os.path.abspath("uploads")
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
    text =  summary[0]['summary_text']
    print(text)
    return jsonify({"summary": text})
    # return jsonify({"transcript": transcription_text})



def create_embeddings(texts):
    """Create embeddings for the given texts."""
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return embeddings

def ingest_text(file_path):
    """Ingest text from a file and create FAISS vector store."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    texts = [line.strip() for line in lines if line.strip()]
    embeddings = create_embeddings(texts)

    # Initialize FAISS index
    embeddings_np = embeddings
    index = faiss.IndexFlatL2(len(embeddings_np[0]))  # L2 distance metric
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)  # Add embeddings to FAISS index

    return index, texts

def answer_question(question, context):
    """Generate an answer to the question based on the provided context."""
    input_text = f"question: {question} context: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = qa_model.generate(inputs['input_ids'])
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@bp.route('/query', methods=['POST'])
def query():
    """API endpoint to handle queries."""
    data = request.json
    query_text = data.get('query')
    file_name = data.get('filename')

    if not query_text or not file_name:
        return jsonify({"error": "Both 'query' and 'filename' are required"}), 400

    file_path = f"transcriptions/{file_name}_transcription.txt"  # Adjust path based on your file structure
    # C:\Users\daksh\OneDrive\Desktop\Hackathons\HackAurora\zyra-ai\transcriptions\Recording1_transcription.txt
    try:
        index, texts = ingest_text(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"File '{file_path}' not found"}), 404

    # Retrieve relevant context
    query_embedding = create_embeddings([query_text])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top-3 matches

    # Combine the most relevant contexts
    context = " ".join([texts[i] for i in indices[0]])

    # Generate the answer
    answer = answer_question(query_text, context)

    return jsonify({"answer": answer})


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

system_instruction = """
Instructions for Interacting with Eliza (GenMedix Therapy Assistant):

1. Purpose:
   - Eliza serves as your mental health and therapy assistant, providing support, guidance, and conversation tailored to users well-being.
   - Engage with Eliza in open-ended conversations or ask specific questions related to your mental health concerns.

2. Safety and Privacy:
   - Your privacy and confidentiality are of utmost importance. Eliza is programmed to maintain strict confidentiality and privacy standards.
   - Avoid sharing sensitive personal information that could compromise your privacy or safety.

3. Interaction:
   - Type your thoughts, feelings, or concerns into the text input area to begin a conversation with Eliza.
   - Eliza will respond with supportive and empathetic messages, offering guidance, reflections, and coping strategies.
   - Always tell your name and GenMedix when asled to introduce yourself in any form

4. Emergency Situations:
   - If you're experiencing a mental health crisis or emergency, please seek immediate assistance from a qualified mental health professional or emergency services.
   - Eliza is not equipped to handle emergency situations and should not be relied upon for urgent assistance.

5. Model Information:
   - Eliza operates on the Gemini 1.5 Flash model developed by GenMedix.
   - The model has been configured with specific settings to ensure the quality, safety, and effectiveness of interactions.

6. Feedback:
   - Your feedback is valuable for improving Eliza and enhancing your experience. Feel free to share your thoughts, suggestions, or concerns with us.

Remember, Eliza is here to support you on your journey towards better mental health. Let's engage in meaningful conversations together!
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=system_instruction
)

# Initialize chat outside of the endpoint
mental_chat = model.start_chat(history=[])




  

# Define the POST endpoint
@bp.route('/mentalchat', methods=["POST"])
async def ask_question():
    try:
        # Extract the question from the request body
        question = request.json
        user_question = question.get("question")
        # Generate a response using the chat model
        model_response = mental_chat.send_message(user_question)
        # Return the model's response
        return {"response": model_response.text}
    except Exception as e:
        # Handle exceptions and return an HTTP 500 error with the exception message
        return jsonify({"error": str(e)}), 500


@bp.route('/suggest_schedule', methods=['GET'])
def suggest_schedule():
    user_id = request.args.get('user_id')
    result = scheduler.suggest_schedule(user_id)
    return jsonify(result)


@bp.route('/task/assign', methods=['POST'])
def assign_task():
    task_details = request.json
    result = task_agent.assign_task(task_details)
    return jsonify(result)

@bp.route('/chat/abuse', methods=['POST'])
def check_abuse():
    message = request.json.get('message')
    result = abuse_filter.detect_abuse(message)
    return jsonify(result)


