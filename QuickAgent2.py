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

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
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

    def save_conversation(self, user_text, ai_text):
        with open("conversation_history.txt", "a") as file:
            file.write(f"User: {user_text}\nAI: {ai_text}\n")

    def load_last_three_exchanges(self):
        try:
            with open("conversation_history.txt", "r") as file:
                lines = file.readlines()
                return "".join(lines[-6:])  # Last 3 exchanges, each has 2 lines (user and AI)
        except FileNotFoundError:
            return ""

    def check_done_talking(self, text):
        previous_conversation = self.load_last_three_exchanges()

        prompt_check = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_check),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessagePromptTemplate.from_template(previous_conversation),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        conversation_check = LLMChain(
            llm=self.llm,
            prompt=prompt_check,
            memory=self.memory
        )

        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()
        response = conversation_check.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM Check ({elapsed_time}ms): {response['text']}")
        return response['text']

    def get_main_response(self, text, session_id):
        api_url = BACKEND_API_URL

        user_message = {
            "role": "user",
            "content": text,
            "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
            "original_list": [],
            "refts": ""
        }

        self.conversation_history.append(user_message)

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
                self.save_conversation(text, assistant_message["content"])  # Save the conversation

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
        return f"<say-as interpret-as='characters'>{grouped_number}</say-as>"

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
                Engine="neural",
                OutputFormat='mp3',
                VoiceId='Joanna'
            )

            if 'AudioStream' in response:
                audio_stream = response['AudioStream']
                with open('speech.mp3', 'wb') as file:
                    file.write(audio_stream.read())
                    audio_stream.close()

                os.system('ffplay -autoexit -nodisp speech.mp3')

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

async def get_transcript(callback):
    try:
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

        await asyncio.Event().wait()  # Keep the function running

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""
        self.silence_start_time = None
        self.pending_transcription = ""  # Buffer for pending transcriptions

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
            self.pending_transcription += " " + full_sentence  # Append new input to pending transcription

        session_id = "sw-1-8f7cf768-9802-4e41-8877-bda08b738959-sti"  # Replace with actual session ID

        # Play greeting message
        self.play_greeting()

        async def check_done_talking():
            while True:
                if self.transcription_response:
                    done_talking_response = self.llm.check_done_talking(self.pending_transcription.strip())
                    
                    if "customer done talking" in done_talking_response.lower():
                        print(f"Customer Query (Complete): {self.pending_transcription.strip()}")  # Log the complete customer query
                        
                        main_responses = self.llm.get_main_response(self.pending_transcription.strip(), session_id)
                        
                        tts = TextToSpeech()
                        for response in main_responses:
                            print(f"LLM Response: {response}")  # Print each LLM response
                            tts.speak(response)
                        
                        self.customer_query = ""  # Reset customer_query after getting the main response
                        self.pending_transcription = ""  # Reset pending transcription after processing
                    else:
                        print(f"Customer Query (Appending): {self.pending_transcription.strip()}")  # Log the appended customer query

                    self.transcription_response = ""  # Reset transcription_response

                await asyncio.sleep(1)  # Check every second

        await asyncio.gather(
            get_transcript(handle_full_sentence),
            check_done_talking()
        )

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())