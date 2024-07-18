import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import json
import re
import boto3
import pyaudio

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

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BACKEND_API_URL = os.getenv("BACKEND_API_URL")
AWS_REGION = os.getenv("AWS_REGION")

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

def format_ssml(text):
    # Regular expression to match email addresses
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    # Regular expression to match numbers with more than 4 digits
    number_pattern = re.compile(r'\b\d{5,}\b')

    # Replace email addresses with SSML for spelling characters
    def replace_email_with_ssml(match):
        email = match.group(0)
        return f"<say-as interpret-as='characters'>{email}</say-as>"

    # Replace numbers with SSML for grouping digits
    def replace_number_with_ssml(match):
        number = match.group(0)
        grouped_number = ' '.join([number[i:i+2] for i in range(0, len(number), 2)])
        return f"<say-as interpret-as='digits'>{grouped_number}</say-as>"

    ssml_text = email_pattern.sub(replace_email_with_ssml, text)
    ssml_text = number_pattern.sub(replace_number_with_ssml, ssml_text)
    return f"<speak>{ssml_text}</speak>"

class TextToSpeech:
    def __init__(self):
        self.polly = boto3.client('polly', region_name=AWS_REGION)

    def speak(self, text):
        ssml_text = format_ssml(text)
        print(f"format_ssml : {ssml_text}")  # Debugging print statement
        try:
            response = self.polly.synthesize_speech(
                Text=ssml_text,
                TextType='ssml',
                Engine="generative",
                OutputFormat='mp3',  # Change to mp3
                VoiceId='Ruth'  # You can choose other voices available in AWS Polly
            )

            if 'AudioStream' in response:
                audio_stream = response['AudioStream']
                with open('speech.mp3', 'wb') as file:
                    file.write(audio_stream.read())
                    audio_stream.close()

                # Play the audio using a system command, suppressing output
                subprocess.run(
                    ['ffplay', '-autoexit', '-nodisp', 'speech.mp3'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

        except Exception as e:
            print(f"Error in Polly TTS: {e}")

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

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, callback, transcript_collector, output_stream, transcription_complete):
        super().__init__(output_stream)
        self.callback = callback
        self.transcript_collector = transcript_collector
        self.transcription_complete = transcription_complete

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                # Process only final results
                for alt in result.alternatives:
                    sentence = alt.transcript
                    self.transcript_collector.add_part(sentence)
                    full_sentence = self.transcript_collector.get_full_transcript()
                    if full_sentence.strip():
                        print(f"Human: {full_sentence.strip()}")
                        self.callback(full_sentence.strip())  # Call the callback with the full sentence
                        self.transcript_collector.reset()  # Reset the collector after processing
                        self.transcription_complete.set()  # Signal transcription completion
                        return  # Stop after the full sentence is handled

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # Setup AWS Transcribe Streaming Client
        client = TranscribeStreamingClient(region=AWS_REGION)

        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        async def write_chunks():
            p = pyaudio.PyAudio()
            audio_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

            try:
                while not transcription_complete.is_set():
                    data = audio_stream.read(1024)
                    await stream.input_stream.send_audio_event(audio_chunk=data)
            except asyncio.CancelledError:
                pass
            finally:
                await stream.input_stream.end_stream()
                audio_stream.stop_stream()
                audio_stream.close()
                p.terminate()

        print("Listening...")  # Print "Listening..." when starting to listen
        # Instantiate our handler and start processing events
        handler = MyEventHandler(callback, transcript_collector, stream.output_stream, transcription_complete)
        await asyncio.gather(write_chunks(), handler.handle_events())

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
