import speech_recognition as sr

def is_english(text):
    # Function to check if a text contains English words
    english_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return any(char in english_characters for char in text)

def main():
    r = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Please say something")

            audio = r.listen(source)  # Set a timeout to limit how long it listens

            if audio:
                print("Recognizing Now .... ")

                try:
                    recognized_text = r.recognize_google(audio, language="en-US", show_all=False)
                    if is_english(recognized_text):
                        print("You said:", recognized_text)

                except sr.UnknownValueError:
                    pass  # Keep it blank and continue


if __name__ == "__main__":
    main()
