import speech_recognition as sr

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
    transcript = []

    # Configure speech recognition parameters
    recognizer.pause_threshold = 0.8  # seconds of non-speaking to end a phrase
    recognizer.energy_threshold = 4000  # adjust based on your microphone sensitivity

    try:
        while True:
            with microphone as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            try:
                text = recognizer.recognize_google(
                    audio,
                    language=language_code
                )
                transcript.append(text)
                print(f"\n[Transcription]: {' '.join(transcript)}", end='\r')
                
            except sr.UnknownValueError:
                print("[Status] Could not understand audio segment", end='\r')
            except sr.RequestError:
                print("[Error] API request failed. Check internet connection.")
                break

    except KeyboardInterrupt:
        print("\n\n** Transcription Stopped **")
        print(f"Final Transcript:\n{' '.join(transcript)}")

if __name__ == "__main__":
    live_bangla_transcription()