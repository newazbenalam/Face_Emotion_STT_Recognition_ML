import speech_recognition as sr
import google.generativeai as genai

# Gemini API configuration
API_KEY = 'YOUR_GEMINI_API_KEY'  # Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

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
        while True:
            with microphone as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            try:
                text = recognizer.recognize_google(
                    audio,
                    language=language_code
                )
                full_transcript.append(text)
                print(f"\n[Transcription]: {' '.join(full_transcript)}", end='\r')
                
                # Send to Gemini API
                if text.strip():
                    response = model.generate_content(
                        f"Process this Bangla text: {text}",
                        stream=True
                    )
                    print("\n\n** Gemini Response **")
                    for chunk in response:
                        print(chunk.text, end='')
                    print("\n" + "="*50 + "\n")

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