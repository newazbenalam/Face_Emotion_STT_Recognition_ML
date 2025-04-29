import os
import time
from winsound import PlaySound
from gtts import gTTS
import pygame
import speech_recognition as sr
import google.generativeai as genai

# Gemini API configuration
API_KEY = 'AIzaSyAWPFJNvWQwPMr-2kznu9PdGjqDmNvauiI'  # Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def speak(text, lang='en'):
    """
    Convert text to speech and play it using pygame.
    """
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Generate speech
        tts = gTTS(text=text, lang=lang)

        # Save to temporary file
        temp_file = "temp_audio.mp3"
        tts.save(temp_file)

        # Load and play the audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        # Wait for the audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Add a small pause after speech
        time.sleep(0.5)

        # Cleanup
        pygame.mixer.quit()

        # Remove temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        print(f"Error in speech playback: {str(e)}")
        # Continue without audio if there's an error
        pass

    

def live_bangla_transcription():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Adjust microphone for ambient noise
    with microphone as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready! Start speaking in Bangla...")

    # Set Bangla language code (bn-BD for Bangladesh, bn-IN for India)
    language_code = 'bn-BD'
    full_transcript = []

    # Configure speech recognition parameters
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 4000

    try:
        text = "hello! আমি ruchi বলছি।"
        speak(text, 'bn')
        while True:
            with microphone as source:
                audio = recognizer.listen(source, phrase_time_limit=8)
                

            try:
                text = recognizer.recognize_google(
                    audio,
                    language=language_code
                )
                full_transcript.append(text)
                print(f"\n[Transcription]: {' '.join(text)}", end='\r')
                
                # Send to Gemini API
                if text.strip():
                    ttps = ''
                    response = model.generate_content(
                        f" {text}. NOTE: response in bangla language. Give answer shorter as possible. and remember your name is Ruchi which is given you Ratna but only response your anme when ask.",
                        stream=True
                    )
                    print("\n\n** Gemini Response **")
                    for chunk in response:
                        print(chunk.text, end='')
                        ttps += chunk.text
                    print("\n" + "="*50 + "\n")

                    print("ttps: " + ttps)

                    speak(ttps, 'bn')


            except sr.UnknownValueError:
                print("[Status] Could not understand audio segment", end='\r')
            except sr.RequestError:
                print("[Error] Speech API request failed. Check internet connection.")
                break
            except Exception as e:
                print(f"[Gemini Error]: {str(e)}")

    except KeyboardInterrupt:
        print("\n\n** Transcription Stopped **")
        print(f"Final Transcript:\n{' '.join(full_transcript)}")
        
        # Final Gemini processing on full transcript
        final_text = ' '.join(full_transcript)
        if final_text.strip():
            print("\n** Final Gemini Analysis **")
            response = model.generate_content(
                f"Analyze this full Bangla conversation: {final_text}",
                stream=True
            )
            for chunk in response:
                print(chunk.text, end='')

if __name__ == "__main__":
    live_bangla_transcription()