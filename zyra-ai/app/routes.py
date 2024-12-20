from flask import Blueprint, request, jsonify
from app.utils import (
    transcription, task_agent, qna_bot, abuse_filter, mental_bot, scheduler, embeddings
)

bp = Blueprint('api', __name__)

@bp.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    language = request.form.get('language', 'en')
    result = transcription.transcribe_audio(audio_file, language)
    return jsonify(result)

@bp.route('/task/assign', methods=['POST'])
def assign_task():
    task_details = request.json
    result = task_agent.assign_task(task_details)
    return jsonify(result)

@bp.route('/qna', methods=['POST'])
def contextual_qna():
    question = request.json.get('question')
    context = request.json.get('context')
    result = qna_bot.answer_question(question, context)
    return jsonify(result)

@bp.route('/chat/abuse', methods=['POST'])
def check_abuse():
    message = request.json.get('message')
    result = abuse_filter.detect_abuse(message)
    return jsonify(result)

@bp.route('/mental_bot', methods=['POST'])
def mental_health_chat():
    user_input = request.json.get('message')
    result = mental_bot.respond(user_input)
    return jsonify(result)

@bp.route('/suggest_schedule', methods=['GET'])
def suggest_schedule():
    user_id = request.args.get('user_id')
    result = scheduler.suggest_schedule(user_id)
    return jsonify(result)
