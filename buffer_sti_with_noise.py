import collections
import sys
import signal
import time
import pyaudio
import webrtcvad
import noisereduce as nr
import numpy as np
import os
import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import json

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

class AudioStream:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      frames_per_buffer=320)

    def read(self):
        try:
            frame = self.stream.read(320, exception_on_overflow=False)
            return frame
        except IOError as e:
            print(f"Error reading from stream: {e}")
            return b''

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

def reduce_noise(audio_frame, sample_rate):
    # Convert bytes to numpy array for noise reduction
    audio_array = np.frombuffer(audio_frame, dtype=np.int16)
    reduced_noise = nr.reduce_noise(y=audio_array, sr=sample_rate)
    return reduced_noise.tobytes()

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, audio_stream):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    start_time = time.time()

    while True:
        frame = audio_stream.read()
        if len(frame) == 0:
            continue  # Skip empty frames resulting from overflow errors

        noise_reduced_frame = reduce_noise(frame, sample_rate)

        try:
            is_speech = vad.is_speech(noise_reduced_frame, sample_rate)
        except webrtcvad.VadError as e:
            print(f"Error processing frame: {e}")
            continue

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
                start_time = None  # Reset timer on speech detection
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f for f in voiced_frames])
                ring_buffer.clear()
                start_time = time.time()  # Start timer on silence detection

        # Check for silence duration
        if not triggered and start_time:
            yield None  # Indicate silence

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
            "is_customer": 0,
            "hash": "e0a507831719956753",
            "site_id": "sti",
            "prompt": "system_role_test_voice",
            "llm_type": "gpt-4o-2024-05-13",
            "url_query": "w%3D440px%26h%3D580px%26user_id%3D%26brand%3Dsw%26site_id%3Dsti%26is_customer%3D0%26region%3Dus%26debug%3D0%26crm_customer_url%3Dhttps%253A%252F%252Fwww.simplytoimpress.com%252F%26start_page_name%3DSimply%2Bto%2BImpress%2B%257C%2BBirth%2BAnnouncements%252C%2BInvitations%252C%2BHoliday%2BCards%26t%3D477766%26llm_type%3Dgpt-4o-2024-05-13"
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

    # Initialize VAD components
    vad_sensitivity = 3  # 0: least aggressive, 3: most aggressive
    vad = webrtcvad.Vad(vad_sensitivity)
    audio_stream = AudioStream()
    sample_rate = 16000
    frame_duration_ms = 20
    padding_duration_ms = 300

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
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

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        silence_start_time = None
        said_hello = False  # Flag to track if "Hello, are you still there?" has been said

        for audio_chunk in vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, audio_stream):
            if audio_chunk:
                silence_start_time = None
                said_hello = False  # Reset the flag when voice activity is detected
                print(f"[{time.strftime('%H:%M:%S')}] Voice activity detected.")
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                else:
                    silence_duration = int(time.time() - silence_start_time)  # Cast to int to show only whole seconds
                    print(f"[{time.strftime('%H:%M:%S')}] Silence duration: {silence_duration} seconds")
                    if 4 < silence_duration <= 9 and not said_hello:
                        print("Hello, are you still there?")
                        os.system('say "Hello, are you still there?"')
                        said_hello = True  # Set the flag to True after saying "Hello, are you still there?"
                    elif silence_duration > 9:
                        print("Cancelling call, no voice activity.")
                        os.system('say "Cancelling call, no voice activity."')
                        break

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""

    def play_greeting(self):
        # Play the pre-recorded greeting message
        greeting_file = "greeting.wav"  # Path to your pre-recorded greeting audio file
        if os.path.exists(greeting_file):
            player_command = ["ffplay", "-autoexit", "-nodisp", greeting_file]
            subprocess.run(player_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Greeting file not found")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        session_id = "sw-1-8f7cf768-9802-4e41-8877-bda08b738959-sti"  # Replace with actual session ID

        # Play greeting message
        self.play_greeting()

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check if the customer is done talking
            done_talking_response = self.llm.check_done_talking(self.transcription_response)
            
            if "customer done talking" in done_talking_response.lower():
                # Append the final part of the transcription to the customer query
                self.customer_query += " " + self.transcription_response
                print(f"Customer Query (Complete): {self.customer_query.strip()}")  # Log the complete customer query
                
                # Get the main response from the backend API
                main_responses = self.llm.get_main_response(self.customer_query.strip(), session_id)
                
                tts = TextToSpeech()
                for response in main_responses:
                    print(f"LLM Response: {response}")  # Print each LLM response
                    tts.speak(response)
                
                # Reset customer_query after getting the main response
                self.customer_query = ""
            else:
                # Append the customer's query until they are done talking
                self.customer_query += " " + self.transcription_response
                print(f"Customer Query (Appending): {self.customer_query.strip()}")  # Log the appended customer query

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())