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
whispermodel = whisper.load_model("base")  # Choose your Whisper model size
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
qa_tokenizer = AutoTokenizer.from_pretrained("t5-small")


from app.utils.transciption import translate
from app.utils.transcribe import  generate_response
from app.utils.prompts import prompt_template, timeline_prompt_template, scheduler_prompt
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
    transcription_result =   whispermodel.transcribe(audio_path)
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
    text = translate(audio_file)
    return jsonify({"summary": text})
    # return jsonify({"transcript": transcription_text})




@bp.route('/query', methods=['POST'])
def query():
    """API endpoint to handle queries."""
    data = request.json
    query_text = data.get('query')
    file_name = data.get('filename')
    file_path = f"transcriptions/{file_name}_transcription.txt"  # Adjust path based on your file structure
    # C:\Users\daksh\OneDrive\Desktop\Hackathons\HackAurora\zyra-ai\transcriptions\Recording1_transcription.txt
 
    if not query_text or not file_name:
        return jsonify({"error": "Both 'query' and 'filename' are required"}), 400

    query_answer = generate_response(query_text, file_name)   

    return jsonify({"answer": query_answer})


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
Instructions for Interacting with Eliza (Zyra Therapy Assistant):

1. Purpose:
   - Zyra serves as your mental health and therapy assistant, providing support, guidance, and conversation tailored to users well-being.
   - Engage with Zyra in open-ended conversations or ask specific questions related to your mental health concerns.

2. Safety and Privacy:
   - Your privacy and confidentiality are of utmost importance. Zyra is programmed to maintain strict confidentiality and privacy standards.
   - Avoid sharing sensitive personal information that could compromise your privacy or safety.

3. Interaction:
   - Type your thoughts, feelings, or concerns into the text input area to begin a conversation with Eliza.
   - Zyra will respond with supportive and empathetic messages, offering guidance, reflections, and coping strategies.
   - Always tell your name and GenMedix when asled to introduce yourself in any form

4. Emergency Situations:
   - If you're experiencing a mental health crisis or emergency, please seek immediate assistance from a qualified mental health professional or emergency services.
   - Zyra is not equipped to handle emergency situations and should not be relied upon for urgent assistance.

5. Model Information:
   - Zyra operates on the Gemini 1.5 Flash model developed by Zyra.
   - The model has been configured with specific settings to ensure the quality, safety, and effectiveness of interactions.
   -It returns answer in 100 words.

6. Feedback:
   - Your feedback is valuable for improving Zyra and enhancing your experience. Feel free to share your thoughts, suggestions, or concerns with us.

Remember, Zyra is here to support you on your journey towards better mental health. Let's engage in meaningful conversations together!
"""



# Initialize Gemini model
modelllm = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=system_instruction
)


# Initialize chat outside of the endpoint
mental_chat = modelllm.start_chat(history=[])




  

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






# Initialize the Gemini model (or any other LLM you prefer)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
chat = gemini_model.start_chat(history=[])



@bp.route("/task-question", methods=["POST"])
async def task_question():
    try:
        # Extract the task and question from the request body
        data = request.json
        task = data.get("task")
        question = data.get("question")

        if not task or not question:
            return jsonify({"error": "Task and question must be provided."}), 400

        # Extract task details
        project = task.get("project", {})
        name = project.get("name", "Unknown Project")
        description = project.get("description", "No description provided.")

        task_details = task.get("task", {})
        taskname = task_details.get("taskname", "Unknown Task")
        task_description = task_details.get("task_description", "No task description provided.")
        due_date = task_details.get("due_date", "No due date specified.")
        status = task_details.get("status", "Status unknown.")

        # Generate the prompt using the template
        prompt = prompt_template.format(
            name=name,
            description=description,
            taskname=taskname,
            task_description=task_description,
            due_date=due_date,
            status=status,
            question=question
        )
        # set up the model amd return the response 
        # Return the response to the user
        response = chat.send_message(prompt).text
        return jsonify({"response": response}), 200

    except Exception as e:
        # Handle exceptions and return an HTTP 500 error with the exception message
        return jsonify({"error": str(e)}), 500



@bp.route('/timeline-task', methods=['POST'])
async def timeline_task():
    try:
        data = request.json
        task = data.get("task")
        # Extract task details
        project = task.get("project", {})
        name = project.get("name", "Unknown Project")
        description = project.get("description", "No description provided.")

        task_details = task.get("task", {})
        taskname = task_details.get("taskname", "Unknown Task")
        task_description = task_details.get("task_description", "No task description provided.")
        due_date = task_details.get("due_date", "No due date specified.")
        status = task_details.get("status", "Status unknown.")

        # Generate the prompt using the template
        prompt = timeline_prompt_template.format(
            name=name,
            description=description,
            taskname=taskname,
            task_description=task_description,
            due_date=due_date,
            status=status,
        )
        # set up the model amd return the response 
        # Return the response to the user
        response = gemini_model.generate_content(prompt)
        print(response)
        return jsonify({"response": response.text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/scheduler', methods=['POST'])
async def scheduler():
    try:
        data = request.json
        users = data.get("users")
        usernames = [user.get("name") for user in users]

        if not users:
            return jsonify({"error": "No users provided."}), 400
        # Extract the timezones of the users
        timezones = [user["timezone"] for user in users]
        # print(timezones)
        # Generate the prompt using the template
        # set up the model amd return the response
        response = gemini_model.generate_content(scheduler_prompt)
        return jsonify({"response": response.text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        