import speech_recognition as sr
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(user_input):
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    return res


def main():
    r = sr.Recognizer()

    print("Chatbot: Hello! Choose an option:")
    print("1. Voice Input")
    print("2. Typed Input")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        while True:
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    print("Please say something:")

                    audio = r.listen(source)  # Adjust the timeout as needed

                    if audio:
                        try:
                            recognized_text = r.recognize_google(audio, language="en-US")
                            print("You said:", recognized_text)
                            chatbot_res = chatbot_response(recognized_text)
                            print("Bot:", chatbot_res)
                        except sr.UnknownValueError:
                            pass
            except sr.WaitTimeoutError:
                pass

    elif choice == '2':
        while True:
            user_input = input("You: ")

            if user_input.lower() == 'exit':
                break

            chatbot_res = chatbot_response(user_input)
            print("Bot:", chatbot_res)

    elif choice == '3':
        return


if __name__ == "__main__":
    main()


