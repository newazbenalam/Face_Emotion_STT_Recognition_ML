import os
import time
import threading
import cv2
import numpy as np
from keras.models import model_from_json
from gtts import gTTS
import pygame
import speech_recognition as sr
import google.generativeai as genai

# Gemini API configuration
API_KEY = 'AIzaSyAWPFJNvWQwPMr-2kznu9PdGjqDmNvauiI'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Shared variables
frame = None
current_emotion = 'neutral'
face_detected = False
running = True
last_interaction = time.time()

def speak(text, lang='bn'):
    """Text-to-speech"""
    try:
        pygame.mixer.init()
        tts = gTTS(text=text, lang=lang)
        tts.save("temp.mp3")
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("temp.mp3")
    except Exception as e:
        print(f"[TTS Error] {e}")
        pygame.mixer.quit()

def emotion_detector():
    """Face detection + emotion prediction"""
    global frame, current_emotion, face_detected, running

    try:
        with open("facialemotionmodel.json", "r") as f:
            emotion_model = model_from_json(f.read())
        emotion_model.load_weights("facialemotionmodel.h5")

        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        cam = cv2.VideoCapture(0)
        time.sleep(2)

        while running and cam.isOpened():
            ret, raw_frame = cam.read()
            if not ret:
                continue
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            face_detected = len(faces) > 0
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0
                pred = emotion_model.predict(roi)
                current_emotion = labels[int(np.argmax(pred))]
                cv2.rectangle(raw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(raw_frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            frame = raw_frame
            time.sleep(0.01)
    except Exception as e:
        print(f"[Emotion Detection Error] {e}")
    finally:
        cam.release()

def auto_respond():
    """Use Gemini API to ask question or give story"""
    try:
        prompt = (
            f"User emotion: {current_emotion}. "
            "Give a short Bangla response and ask the user if they want to hear a story. "
            "Speak in Bangla only. Mention your name 'রুচি' if asked."
        )
        response = model.generate_content(prompt)
        message = response.text.strip()
        print(f"[Gemini]: {message}")
        speak(message, 'bn')
    except Exception as e:
        print(f"[Gemini Error] {e}")

def transcription_loop():
    global running, last_interaction

    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("[Speech] Calibrating...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("[Speech] Ready.")

    speak("হ্যালো! আমি রুচি বলছি।", 'bn')
    auto_respond()

    while running:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=5)
            text = r.recognize_google(audio, language='bn-BD')
            print(f"[User]: {text} | Emotion: {current_emotion}")

            prompt = (
                f"User emotion and mood: {current_emotion}. "
                f"User message: {text}. "
                "Respond in Bangla. Keep answers short but when you are telling a story keep it moderate. "
                "Only mention your name 'রুচি' when asked. You are developed and created by Ratna."
            )
            response = model.generate_content(prompt, stream=True)
            response_text = ''.join([chunk.text for chunk in response])
            print("\n** Gemini Response **")
            print(response_text)
            print("="*50)
            speak(response_text, 'bn')
                
        except sr.UnknownValueError:
            print("[Speech] Didn't understand")
        except Exception as e:
            print(f"[Speech Error] {e}")

def show_video():
    """Main GUI loop to show frame"""
    global running
    while running:
        if frame is not None:
            try:
                cv2.imshow("Emotion Detection (press 'x' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    running = False
                    break
            except Exception as e:
                print(f"[Display Error] {e}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # macOS fix

    thread_emotion = threading.Thread(target=emotion_detector, daemon=True)
    thread_emotion.start()

    thread_speech = threading.Thread(target=transcription_loop)
    thread_speech.start()

    show_video()  # Run in main thread

    running = False
    print("Shutting down...")
