import asyncio
import os
import queue
import time
import pyaudio
import subprocess
from google.cloud import speech, texttospeech
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Load environment variables from .env file for GROQ_API_KEY
load_dotenv()

# Set the path to the service account key directly
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("sa_planetart-593fa47584a4.json")

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class GoogleCloudSpeechToText:
    def __init__(self):
        self.client = speech.SpeechClient()
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=True,
                enable_spoken_emojis=True,
                model="latest_long",
                use_enhanced=True,
            ),
            interim_results=True,
            single_utterance=False,
        )
        self.audio_queue = queue.Queue()

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def listen_print_loop(self, responses, callback):
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            if result.is_final:
                callback(transcript)

    async def start_transcription(self, callback):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback,
        )

        stream.start_stream()

        responses = self.client.streaming_recognize(self.streaming_config, self.stream_generator())
        self.listen_print_loop(responses, callback)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def stream_generator(self):
        while True:
            yield speech.StreamingRecognizeRequest(audio_content=self.audio_queue.get())

class GoogleCloudTextToSpeech:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def speak(self, text, language_code="en-US"):
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open("output.wav", "wb") as out:
            out.write(response.audio_content)

        # Use ffplay to play the audio
        player_command = ["ffplay", "-autoexit", "output.wav", "-nodisp"]
        subprocess.run(player_command)

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        stt = GoogleCloudSpeechToText()

        def on_final_transcript(transcript):
            print(f"Human: {transcript}")
            callback(transcript)
            transcription_complete.set()

        await stt.start_transcription(on_final_transcript)

        await transcription_complete.wait()  # Wait for transcription to complete

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.accumulated_text = []
        self.csv_file = "conversation_log.csv"
        self.stt = GoogleCloudSpeechToText()
        self.tts = GoogleCloudTextToSpeech()
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as file:
                file.write("full_conversation\n")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Append the current transcription to the accumulated text
            self.accumulated_text.append(self.transcription_response)
            
            # Check if the user is done talking
            combined_text = " ".join(self.accumulated_text)
            llm_response = self.llm.process(combined_text)
            
            print(f"Combined Text: {combined_text}")
            print(f"LLM Response: {llm_response}")

            # If LLM determines that the user is done talking
            if "user is done talking" in llm_response.lower():
                # Save the accumulated text as a single entry in the CSV file
                with open(self.csv_file, 'a') as file:
                    file.write(f"{combined_text}\n")
                print(f"Saved to CSV: {combined_text}")

                # Reset the accumulated text
                self.accumulated_text = []
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            self.tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())