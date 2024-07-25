import asyncio
from dotenv import load_dotenv
import threading
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
        self.llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_history = []
        self.last_transcriptions = []
        self.last_assistant_response = ""  # Store the last assistant response
        # Load the system prompts from files
        with open('system_prompt_check.txt', 'r') as file:
            self.system_prompt_check = file.read().strip()

    def clean_numbers(self, text):
        # This regex replaces any non-digit character between digits with nothing
        return re.sub(r'(\d)[^\d]+(\d)', r'\1\2', text)

    def check_done_talking(self, text, exclude_interrupted=False):
        # Add the current transcription to the list
        self.last_transcriptions.append(text)
        # Keep only the last three transcriptions (previous two and the current one)
        if len(self.last_transcriptions) > 7:
            self.last_transcriptions.pop(0)

        # Combine the last three transcriptions into one string
        combined_text = ' '.join(self.last_transcriptions)

        # Clean numbers in the combined text
        cleaned_combined_text = self.clean_numbers(combined_text)

        # Format the conversation history, including only the last user and assistant message
        last_user_message = next((entry for entry in reversed(self.conversation_history) if entry['role'] == 'user'), None)
        last_assistant_message = next((entry for entry in reversed(self.conversation_history) if entry['role'] == 'assistant' and not entry.get('interrupted')), None)
        
        conversation_history = ""
        if last_assistant_message:
            conversation_history += f"Assistant: {last_assistant_message['content']}\n"
        if last_user_message:
            conversation_history += f"User: {last_user_message['content']}\n"
        
        cleaned_combined_text_with_assistant_response = f"{conversation_history}User: {cleaned_combined_text}"
        print(f"Checking if done talking with combined text: {cleaned_combined_text_with_assistant_response}")

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

        self.memory.chat_memory.add_user_message(cleaned_combined_text_with_assistant_response)  # Add combined message to memory

        start_time = time.time()

        # Get the response from the LLM for checking if done talking
        response = conversation_check.invoke({"text": cleaned_combined_text_with_assistant_response})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM Check ({elapsed_time}ms): {response['text']}")

        return response['text']

    def get_main_response(self, text, session_id):
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

        try:
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
                    full_responses.append(assistant_message)
                    # Append new assistant message to conversation history
                    self.conversation_history.append(assistant_message)

                # Update the last assistant response
                if full_responses:
                    self.last_assistant_response = full_responses[-1]["content"]  # Ensure this is a string

                return [message["content"] for message in full_responses]
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response content: {response.content}")
                return ["There was an error processing your request. Please try again later."]
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
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
        self.playback_process = None
        self.playback_active = False

    def play_audio(self, audio_stream):
        with open('speech.mp3', 'wb') as file:
            file.write(audio_stream.read())
            audio_stream.close()

        # Play the audio using a system command, suppressing output
        self.playback_process = subprocess.Popen(
            ['ffplay', '-autoexit', '-nodisp', 'speech.mp3'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self.playback_active = True
        print(f"playback_active set to True")
        self.playback_process.wait()
        self.playback_active = False
        print(f"playback_active set to False")

    def speak(self, text):
        ssml_text = format_ssml(text)
        print(f"format_ssml: {ssml_text}")  # Debugging print statement
        try:
            response = self.polly.synthesize_speech(
                Text=ssml_text,
                TextType='ssml',
                Engine="neural",
                OutputFormat='mp3',
                VoiceId='Ruth'
            )

            if 'AudioStream' in response:
                audio_stream = response['AudioStream']

                # Start a new thread for audio playback
                playback_thread = threading.Thread(target=self.play_audio, args=(audio_stream,))
                playback_thread.start()
                playback_thread.join()  # Wait for the playback to finish

        except Exception as e:
            print(f"Error in Polly TTS: {e}")

    def stop(self):
        if self.playback_process and self.playback_process.poll() is None:
            self.playback_process.terminate()
            self.playback_process.wait()
            self.playback_process = None
            self.playback_active = False
            print(f"playback_active set to False (stopped)")

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
        self.tts = TextToSpeech()
        self.interrupt_event = threading.Event()
        self.playback_finished_event = threading.Event()

    def play_greeting(self):
        greeting_file = "greeting.mp3"
        if os.path.exists(greeting_file):
            player_command = ["ffplay", "-autoexit", "-nodisp", greeting_file]
            subprocess.run(player_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Greeting file not found")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            self.interrupt_event.set()

        session_id = "sw-1-8f7cf768-9802-4e41-8877-bda08b738959-sti"  # Replace with actual session ID

        self.play_greeting()

        while True:
            self.interrupt_event.clear()
            self.playback_finished_event.clear()
            interruption_processed = False

            listen_thread = threading.Thread(target=asyncio.run, args=(get_transcript(handle_full_sentence),))
            listen_thread.start()
            listen_thread.join()

            done_talking_response = self.llm.check_done_talking(self.transcription_response)

            if "customer done talking" in done_talking_response.lower():
                self.customer_query += " " + self.transcription_response
                print(f"Customer Query (Complete): {self.customer_query.strip()}")

                main_responses = self.llm.get_main_response(self.customer_query.strip(), session_id)

                for response in main_responses:
                    print(f"LLM Response: {response}")
                    speak_thread = threading.Thread(target=self.tts.speak, args=(response,))
                    speak_thread.start()

                    while speak_thread.is_alive():
                        listen_thread = threading.Thread(target=asyncio.run, args=(get_transcript(handle_full_sentence),))
                        listen_thread.start()
                        listen_thread.join()

                        if self.transcription_response.strip():
                            if self.tts.playback_active:
                                print("User interrupted during LLM response")
                                self.tts.stop()
                                speak_thread.join()
                                self.customer_query += " " + self.transcription_response
                                self.transcription_response = ""
                                interruption_processed = True
                                break

                    if not self.tts.playback_active:
                        self.playback_finished_event.set()
                        print(f"Playback finished, handling follow-up speech: {self.transcription_response}")
                        if self.transcription_response.strip() and not interruption_processed:
                            self.customer_query += " " + self.transcription_response
                        self.transcription_response = ""
                        interruption_processed = False

                        done_talking_response = self.llm.check_done_talking(self.customer_query.strip())
                        if "customer done talking" in done_talking_response.lower():
                            main_responses = self.llm.get_main_response(self.customer_query.strip(), session_id)
                            for response in main_responses:
                                print(f"LLM Response: {response}")
                                self.interrupt_event.clear()
                                speak_thread = threading.Thread(target=self.tts.speak, args=(response,))
                                speak_thread.start()
                                speak_thread.join()
                                self.interrupt_event.clear()
                            self.customer_query = ""

            else:
                self.customer_query += " " + self.transcription_response
                print(f"Customer Query (Appending): {self.customer_query.strip()}")

            self.transcription_response = ""

            if not self.tts.playback_active and self.playback_finished_event.is_set():
                print(f"Playback finished, handling follow-up speech: {self.transcription_response}")
                if self.transcription_response.strip():
                    self.customer_query += " " + self.transcription_response
                    self.transcription_response = ""
                    done_talking_response = self.llm.check_done_talking(self.customer_query.strip())
                    if "customer done talking" in done_talking_response.lower():
                        main_responses = self.llm.get_main_response(self.customer_query.strip(), session_id)
                        for response in main_responses:
                            print(f"LLM Response: {response}")
                            self.interrupt_event.clear()
                            speak_thread = threading.Thread(target=self.tts.speak, args=(response,))
                            speak_thread.start()
                            speak_thread.join()
                            self.interrupt_event.clear()
                        self.customer_query = ""

            if not self.playback_finished_event.is_set():
                continue

            # Check for exit condition
            if "goodbye" in self.customer_query.lower():
                print("Goodbye detected. Exiting conversation.")
                break

        print("Conversation ended.")

def run_conversation_manager():
    manager = ConversationManager()
    asyncio.run(manager.main())

if __name__ == "__main__":
    run_conversation_manager()
    