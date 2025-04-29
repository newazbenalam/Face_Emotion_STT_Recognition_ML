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
import sounddevice as sd
import soundfile as sf
from animation import display_eyes

def set_eyes(emotion):
    """Change eyes animation safely using threading"""
    threading.Thread(target=display_eyes, args=(emotion,), daemon=True).start()


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

def record_audio(duration=5, samplerate=44100, channels=1, device=None):
    """Record audio using sounddevice"""
    try:
        set_eyes('listening')
        print("Recording...")
        if not device:
            device = None  # Default microphone
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=device)
        sd.wait()
        sf.write('input.wav', recording, samplerate)
        print("Recording saved as input.wav")
        return 'input.wav'
    except Exception as e:
        print(f"Recording Error: {str(e)}")
        return None

def auto_question_and_response(recognizer, microphone, full_transcript, language_code):
    """Handle automatic questioning and response"""
    global running, last_interaction, current_emotion
    try:
        prompt = (
            f"User emotion and mood: {current_emotion}. "
            "Keep answers short and ask user if they want to hear a story depending on their emotion status, in Bangla to engage the user conversational. Always respond in Bangla only. "
        )
        response = model.generate_content(prompt)
        question = response.text.strip()
        print(f"\n** Auto Question **\n{question}")
        speak(question, 'bn')
        last_interaction = time.time()

    except Exception as e:
        print(f"Auto question error: {str(e)}")

def face_detection():
    """Real-time emotion detection thread"""
    global current_emotion, running, face_detected
    retry_count = 0
    frame_skip = 3  # Moved frame_skip initialization outside the loop
    
    try:
        # Load emotion detection model
        json_file = open("facialemotionmodel.json", "r")
        model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(model_json)
        emotion_model.load_weights("facialemotionmodel.h5")
        
        # Verify Haar cascade exists
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_file):
            raise FileNotFoundError(f"Haar cascade not found: {haar_file}")
            
        face_cascade = cv2.CascadeClassifier(haar_file)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        # Initialize webcam
        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)  # Camera warm-up
        
        while running and retry_count < 5:
            ret, frame = webcam.read()
            if not ret:
                print("Frame read failed. Retrying...")
                retry_count += 1
                time.sleep(1)
                continue
            retry_count = 0
            
            try:
                # Skip frames to reduce load
                frame_skip = (frame_skip + 1) % 2
                if frame_skip != 0:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))  # Added minSize filter

                if len(faces) > 0:
                    # Choose the largest face based on area (w*h)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    (x, y, w, h) = largest_face
                    # Continue processing the selected face
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img = roi_gray.reshape(1, 48, 48, 1) / 255.0
                    pred = emotion_model.predict(img)
                    current_emotion = labels[pred.argmax()]
                    
                    # Draw a rectangle around the largest face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, current_emotion, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                
                cv2.imshow('Emotion Detection (x to exit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    running = False
                    break
                    
            except cv2.error as e:
                print(f"OpenCV error: {str(e)}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
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
            if face_detected and (time.time() - last_interaction > 2):
                auto_question_and_response(recognizer, microphone, full_transcript, language_code)

            try:
                # Record audio input
                audio_file = record_audio(duration=5)
                if audio_file:
                    # Process the recorded audio and convert it to text (speech-to-text)
                    with sr.AudioFile(audio_file) as source:
                        audio = recognizer.record(source)
                        text = recognizer.recognize_google(audio, language=language_code)
                        full_transcript.append(text)
                        print(f"\n[Transcription]: {text} | Emotion: {current_emotion}")

                        prompt = (
                            f"User emotion and mood: {current_emotion}. "
                            f"User message: {text}. "
                            "Respond in Bangla. Keep answers short but when you are telling a story keep it moderate. "
                            "Only mention your name 'রুচি' when asked. You are developed and created by Ratna."
                        )
                        response = model.generate_content(prompt, stream=True)
                        response_text = ''.join([chunk.text for chunk in response])
                        last_interaction = time.time()
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
    # set_eyes('neutral')
    # macOS specific fix for Continuity Camera
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    
    emotion_thread = threading.Thread(target=face_detection, daemon=True)
    emotion_thread.start()
    live_bangla_transcription()
    print("Exiting program...")
    cv2.destroyAllWindows()
    time.sleep(1)
