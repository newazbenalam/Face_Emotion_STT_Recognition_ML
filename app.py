import os
import time
import threading
import cv2
from keras.models import model_from_json
import numpy as np
from gtts import gTTS
import pygame
import speech_recognition as sr
import google.generativeai as genai

# Gemini API configuration
API_KEY = 'AIzaSyAWPFJNvWQwPMr-2kznu9PdGjqDmNvauiI'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global control variables
running = True
current_emotion = 'neutral'
face_detected = False
last_interaction = time.time()

def speak(text, lang='en'):
    """Text-to-speech function"""
    try:
        pygame.mixer.init()
        tts = gTTS(text=text, lang=lang)
        temp_file = "temp_audio.mp3"
        tts.save(temp_file)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and running:
            pygame.time.Clock().tick(10)
        time.sleep(0.5)
        pygame.mixer.quit()
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        print(f"TTS Error: {str(e)}")

def auto_question_and_response(recognizer, microphone, full_transcript, language_code):
    """Handle automatic questioning and response"""
    global running, last_interaction, current_emotion
    try:
        prompt = (
            f"User emotion and mood: {current_emotion}. "
            "Generate a interesting question in Bangla to engage the user conversational. Always respond in Bangla only. "
        )
        response = model.generate_content(prompt)
        question = response.text.strip()
        print(f"\n** Auto Question **\n{question}")
        speak(question, 'bn')
        last_interaction = time.time()

        # with microphone as source:
        #     audio = recognizer.listen(source, phrase_time_limit=10)
        # text = recognizer.recognize_google(audio, language=language_code)
        # full_transcript.append(text)
        # print(f"\n[Transcription]: {text}")

        # prompt = (
        #     f"User emotion and mood: {current_emotion}. "
        #     f"User message: {text}. "
        #     "Always respond in Bangla only. Keep answers short. "
        #     "Only mention your name 'রুচি' when asked."
        # )
        # response = model.generate_content(prompt, stream=True)
        # response_text = ''.join([chunk.text for chunk in response])
        # print("\n** Gemini Response **")
        # print(response_text)
        # print("="*50)
        # speak(response_text, 'bn')
        last_interaction = time.time()
    except sr.UnknownValueError:
        print("[Status] No response detected after auto question")
    except Exception as e:
        print(f"Auto question error: {str(e)}")

def face_detection():
    """Real-time emotion detection thread"""
    global current_emotion, running, face_detected
    json_file = open("facialemotionmodel.json", "r")
    model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(model_json)
    emotion_model.load_weights("facialemotionmodel.h5")
    
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    webcam = cv2.VideoCapture(1)
    
    while running and webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img = roi_gray.reshape(1, 48, 48, 1) / 255.0
            pred = emotion_model.predict(img)
            current_emotion = labels[pred.argmax()]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, current_emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection (Press x to exit)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            running = False
            break
            
    webcam.release()
    cv2.destroyAllWindows()

def live_bangla_transcription():
    """Main transcription loop"""
    global running, last_interaction, face_detected
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready! Start speaking in Bangla...")

    language_code = 'bn-BD'
    full_transcript = []
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 4000

    try:
        speak("হ্যালো! আমি রুচি বলছি।", 'bn')
        last_interaction = time.time()
        auto_question_and_response(recognizer, microphone, full_transcript, language_code)
        
        while running:
            if face_detected and (time.time() - last_interaction > 20):
                auto_question_and_response(recognizer, microphone, full_transcript, language_code)

            try:
                with microphone as source:
                    audio = recognizer.listen(source, phrase_time_limit=3)
            except OSError:
                break

            if not running:
                break

            try:
                text = recognizer.recognize_google(audio, language=language_code)
                full_transcript.append(text)
                print(f"\n[Transcription]: {text} | Emotion: {current_emotion}")
                last_interaction = time.time()
                
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
                print("[Status] Could not understand audio segment")
            except Exception as e:
                print(f"Error: {str(e)}")

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n\n** System Shutdown Initiated **")
        time.sleep(1)

if __name__ == "__main__":
    emotion_thread = threading.Thread(target=face_detection, daemon=True)
    emotion_thread.start()
    live_bangla_transcription()
    print("Exiting program...")
    cv2.destroyAllWindows()
    time.sleep(1)