from flask import Flask, render_template, request, jsonify
import json
import os
import speech_recognition as sr
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json["user_input"]
    bot_response = chatbot_response(user_input)
    return jsonify({"response": bot_response})

@app.route("/voice_input", methods=["POST"])
def voice_input():
    audio = request.files.get("audio")
    r = sr.Recognizer()

    with sr.AudioFile(audio) as source:
        try:
            audio_text = r.recognize_google(source)
            bot_response = chatbot_response(audio_text)
            return jsonify({"response": bot_response})
        except sr.UnknownValueError:
            return jsonify({"response": "Sorry, I couldn't understand the audio."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5999, debug=True)