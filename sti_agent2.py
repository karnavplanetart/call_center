import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import json
import webrtcvad
import pyaudio
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_history = []

        # Load the system prompts from files
        with open('system_prompt_check.txt', 'r') as file:
            self.system_prompt_check = file.read().strip()

    def check_done_talking(self, text):
        prompt_check = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_check),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        conversation_check = LLMChain(
            llm=self.llm,
            prompt=prompt_check,
            memory=self.memory
        )

        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Get the response from the LLM for checking if done talking
        response = conversation_check.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM Check ({elapsed_time}ms): {response['text']}")
        return response['text']

    def get_main_response(self, text, session_id):
        # Define the API endpoint and payload
        api_url = BACKEND_API_URL

        # Create a new user message
        user_message = {
            "role": "user",
            "content": text,
            "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
            "original_list": [],
            "refts": ""
        }

        # Append new user message to conversation history
        self.conversation_history.append(user_message)

        # Create payload with the current conversation history
        payload = {
            "messages": self.conversation_history,
            "brand": "sw",
            "user_id": "",
            "session_id": session_id,
            "debug": "0",
            "token": "",
            "is_customer": 1,
            "hash": "e0a507831719956753",
            "site_id": "sti",
            "role": "system_role_test_voice",
            "url_query": "w%3D440px%26h%3D580px%26user_id%3D%26brand%3Dsw%26site_id%3Dsti%26is_customer%3D1%26region%3Dus%26debug%3D0%26crm_customer_url%3Dhttps%253A%252F%252Fwww.simplytoimpress.com%252F%26start_page_name%3DSimply%2Bto%2BImpress%2B%257C%2BBirth%2BAnnouncements%252C%2BInvitations%252C%2BHoliday%2BCards%26t%3D477766"
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Send the POST request to the backend API
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            # Extract the assistant's response content
            assistant_replies = response_data.get("messages", [])

            full_responses = []
            for reply in assistant_replies:
                assistant_message = {
                    "role": "assistant",
                    "content": reply.get("content", ""),
                    "ts": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                full_responses.append(assistant_message["content"])
                # Append new assistant message to conversation history
                self.conversation_history.append(assistant_message)

            return full_responses
        else:
            return ["There was an error processing your request. Please try again later."]

class TextToSpeech:
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time) * 1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

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
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=500,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input
                # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class VoiceActivityDetector:
    def __init__(self, aggressiveness=3):
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio_chunk):
        return self.vad.is_speech(audio_chunk, sample_rate=16000)

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""
        self.last_input_time = time.time()
        self.inactivity_warning_issued = False
        self.vad = VoiceActivityDetector()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(format=pyaudio.paInt16,
                                                 channels=1,
                                                 rate=16000,
                                                 input=True,
                                                 frames_per_buffer=160)  # 10ms frame at 16000 Hz

    def get_audio_chunk(self):
        try:
            data = self.stream.read(160, exception_on_overflow=False)  # 10ms frame
            return data
        except IOError as e:
            print(f"Input overflowed: {e}")
            return b''

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            self.last_input_time = time.time()  # Reset the last input time when user speaks
            self.inactivity_warning_issued = False  # Reset the warning flag

        session_id = "sw-1-8f7cf768-9802-4e41-8877-bda08b738948-sti"  # Replace with actual session ID
        tts = TextToSpeech()  # Initialize TTS once

        while True:
            # Run get_transcript asynchronously
            transcript_task = asyncio.create_task(get_transcript(handle_full_sentence))
            
            # Check for voice activity using VAD in parallel
            while not transcript_task.done():
                audio_chunk = self.get_audio_chunk()
                if len(audio_chunk) == 0:
                    continue

                # Ensure the audio chunk is in the correct format
                audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16).tobytes()

                try:
                    if self.vad.is_speech(audio_chunk):
                        self.last_input_time = time.time()  # Reset the last input time when speech is detected
                except webrtcvad.VadError as e:
                    print(f"VAD error: {e}")
                    continue
                
                current_time = time.time()
                inactivity_duration = current_time - self.last_input_time

                if inactivity_duration >= 10:
                    tts.speak("Cutting the call due to no user activity.")
                    print("Cutting the call due to no user activity.")
                    transcript_task.cancel()
                    break

                if inactivity_duration >= 5 and not self.inactivity_warning_issued:
                    tts.speak("Are you there?")
                    print("Are you there?")
                    self.inactivity_warning_issued = True

                await asyncio.sleep(0.1)  # Add a small delay to prevent a busy loop

            # Await the completion of the transcription task
            try:
                await transcript_task
            except asyncio.CancelledError:
                break

            if self.transcription_response:
                # Check if the customer is done talking
                done_talking_response = self.llm.check_done_talking(self.transcription_response)
                
                if "customer done talking" in done_talking_response.lower():
                    self.customer_query = self.transcription_response
                    print(f"Customer Query (Complete): {self.customer_query.strip()}")  # Log the complete customer query
                    # Get the main response from the backend API
                    main_responses = self.llm.get_main_response(self.customer_query.strip(), session_id)
                    
                    for response in main_responses:
                        print(f"LLM Response: {response}")  # Print each LLM response
                        tts.speak(response)
                    
                    self.customer_query = ""  # Reset customer query after getting the main response
                else:
                    # Append the customer's query until they are done talking
                    self.customer_query += " " + self.transcription_response
                    print(f"Customer Query (Appending): {self.customer_query.strip()}")  # Log the appended customer query

                # Reset transcription_response for the next loop iteration
                self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())