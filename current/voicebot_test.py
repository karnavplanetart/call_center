import asyncio
import threading
import signal
from dotenv import load_dotenv
import shutil
import subprocess
import json
import requests
import time
import webrtcvad
import os
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

import torch
import torchaudio
import threading
import asyncio

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
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
        api_url = os.getenv("BACKEND_API_URL")

        if not api_url:
            print("Error: BACKEND_API_URL environment variable is not set.")
            return ["There was an error processing your request. Please try again later."]

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

        #print(f"Sending request to backend API: {api_url}")
        #print(f"Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            assistant_replies = response_data.get("messages", [])

            full_responses = []
            for reply in assistant_replies:
                assistant_message = {
                    "role": "assistant",
                    "content": reply.get("content", ""),
                    "ts": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                full_responses.append(assistant_message["content"])
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

    def speak(self, text, stop_event):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

        try:
            with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk and not stop_event.is_set():
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()
                    if stop_event.is_set():
                        print("Interrupt detected. Stopping audio playback.")
                        os.killpg(os.getpgid(player_process.pid), signal.SIGTERM)
                        break
        except Exception as e:
            print(f"Error during audio playback: {e}")
        finally:
            if player_process.stdin:
                player_process.stdin.close()
            player_process.wait()
            print("Audio playback process terminated.")


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
    transcription_complete = asyncio.Event()

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

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

        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()

        microphone.finish()

        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class VoiceActivityDetector:
    def __init__(self, mic_device_index):
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils
        self.mic_device_index = mic_device_index
        self.threshold = 0.6

    def detect(self, stop_event, audio_in_progress_event):
        vad_iterator = self.VADIterator(self.model)

        def microphone_callback(data):
            if stop_event.is_set():
                return False
            audio_torch = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            speech_timestamps = self.get_speech_timestamps(audio_torch, self.model)
            
            # Adjust this condition based on your sensitivity requirements
            if speech_timestamps and audio_in_progress_event.is_set():
                stop_event.set()
                return False
            return True

        mic = Microphone(push_callback=microphone_callback, input_device_index=self.mic_device_index)
        mic.start()

        while not stop_event.is_set():
            time.sleep(0.3)

        mic.finish()


class ConversationManager:
    def __init__(self, mic_device_index):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""
        self.mic_device_index = mic_device_index

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

        vad = VoiceActivityDetector(self.mic_device_index)

        def vad_monitor(stop_event, audio_in_progress_event):
            vad.detect(stop_event, audio_in_progress_event)

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
                
                # Initialize VAD monitoring
                stop_event = threading.Event()
                audio_in_progress_event = threading.Event()
                vad_thread = threading.Thread(target=vad_monitor, args=(stop_event, audio_in_progress_event), daemon=True)
                vad_thread.start()

                tts = TextToSpeech()
                audio_in_progress_event.set()
                for response in main_responses:
                    print(f"LLM Response: {response}")
                    tts.speak(response, stop_event)

                vad_thread.join()
                audio_in_progress_event.clear()

                self.customer_query = ""
            else:
                # Append the customer's query until they are done talking
                self.customer_query += " " + self.transcription_response
                print(f"Customer Query (Appending): {self.customer_query.strip()}")  # Log the appended customer query

            self.transcription_response = ""

if __name__ == "__main__":
    load_dotenv()
    mic_device_index = 1  # Replace with the actual index of your microphone device
    manager = ConversationManager(mic_device_index)
    asyncio.run(manager.main())