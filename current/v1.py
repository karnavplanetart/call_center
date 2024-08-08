import asyncio
from dotenv import load_dotenv
import subprocess
import requests
import time
import os
import json
import re
import boto3
import sounddevice as sd
import uuid
from colorama import init, Fore, Style

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

# Initialize colorama
init(autoreset=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BACKEND_API_URL = os.getenv("BACKEND_API_URL")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

transcribe_client = TranscribeStreamingClient(region=AWS_REGION)

def generate_session_id():
    unique_id = uuid.uuid4()
    return f"sw-1-{unique_id}-sti"

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_history = []

        # Load the system prompts from files
        with open('system_prompt_check.txt', 'r') as file:
            self.system_prompt_check = file.read().strip()

    def get_combined_conversation(self, n):
        conv_combined_text = f"{Fore.BLUE}Combined Conversation for LLM Check:\n"
        # Get the last n messages from conversation history
        last_n_messages = self.conversation_history[-n:]
        
        for message in last_n_messages:
            role = "User" if message["role"] == "user" else "Assistant"
            conv_combined_text += f"{role}: {message['content']}\n"
        
        conv_combined_text += f"{Style.RESET_ALL}"
        return conv_combined_text.strip()

    def check_done_talking(self, text, n=5):
        combined_conversation = self.get_combined_conversation(n)
        combined_conversation += f"{Fore.BLUE}\nUser: {text}{Style.RESET_ALL}"  # Add the current user message

        # Print the combined conversation text for clarity
        print(combined_conversation)

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
        response = conversation_check.invoke({"text": combined_conversation})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        if "not_done_talking" in response['text'].lower():
            print(f"{Fore.YELLOW}LLM Check ({elapsed_time}ms): {response['text']}{Style.RESET_ALL}")
        elif "customer_done_talking" in response['text'].lower():
            print(f"{Fore.GREEN}LLM Check ({elapsed_time}ms): {response['text']}{Style.RESET_ALL}")
        else:
            print(f"LLM Check ({elapsed_time}ms): {response['text']}")
        return response['text']

    async def get_main_response(self, text, session_id, query_id, interrupt_flag):
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
            "url_query": "w%3D440px%26h%3D580px%26user_id%3D%26brand%3Dsw%26site_id%3Dsti%26is_customer%3Dus%26debug%3D0%26crm_customer_url%3Dhttps%253A%252F%252Fwww.simplytoimpress.com%252F%26start_page_name%3DSimply%2Bto%2BImpress%2B%257C%2BBirth%2BAnnouncements%252C%2BInvitations%252C%2BHoliday%2BCards%26t%3D477766%26llm_type%3Dgpt-4o-2024-05-13"
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Send the POST request to the backend API
            response = await asyncio.to_thread(requests.post, api_url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                response_data = response.json()
                # Extract the assistant's response content
                assistant_replies = response_data.get("messages", [])

                full_responses = []
                for reply in assistant_replies:
                    # Check if interrupted
                    if interrupt_flag.is_set():
                        print("Processing interrupted by customer")
                        return []

                    assistant_message = {
                        "role": "assistant",
                        "content": reply.get("content", ""),
                        "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "query_id": query_id  # Attach the unique identifier to the assistant message
                    }
                    full_responses.append(assistant_message)

                    # Append new assistant message to conversation history
                    self.conversation_history.append(assistant_message)

                return full_responses
            else:
                return [{"content": "There was an error processing your request. Please try again later.", "query_id": query_id}]
        except Exception as e:
            print(f"Error during get_main_response: {e}")
            return [{"content": "There was an error processing your request. Please try again later.", "query_id": query_id}]

class TextToSpeech:
    def __init__(self):
        self.polly = boto3.client('polly', region_name=AWS_REGION)
        self.playing = False
        self.process = None

    async def speak(self, text):
        def format_ssml(text):
            # Regular expression to match email addresses
            email_pattern = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
            # Regular expression to match numbers with more than 4 digits
            number_pattern = re.compile(r'\b\d{5,}\b')
            # Function to replace email with SSML
            def replace_email_with_ssml(match):
                local_part = match.group(1)
                domain_part = match.group(2)
                return f"<break time='300ms'/><say-as interpret-as='characters'>{local_part}</say-as><break time='300ms'/>@{domain_part}"
            # Function to replace number with SSML
            def replace_number_with_ssml(match):
                number = match.group(0)
                grouped_number = ' '.join([number[i:i+2] for i in range(0, len(number), 2)])
                return f"<say-as interpret-as='telephone'>{grouped_number}</say-as>"
            ssml_text = email_pattern.sub(replace_email_with_ssml, text)
            ssml_text = number_pattern.sub(replace_number_with_ssml, ssml_text)
            return f"<speak>{ssml_text}</speak>"
        
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

                # Play the audio using a system command
                self.playing = True
                print("TTS playback started")
                self.process = await asyncio.create_subprocess_exec(
                    'ffplay', '-autoexit', '-nodisp', 'speech.mp3',
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await self.process.communicate()
                self.playing = False
                print("TTS playback stopped")

        except Exception as e:
            self.playing = False
            print(f"Error in Polly TTS: {e}")

    async def stop(self):
        if self.playing and self.process:
            self.process.terminate()
            await self.process.wait()
            self.playing = False
            print("TTS playback stopped by interruption")

    async def speak_timeout_message(self, message):
        await self.speak(message)

    async def play_audio_cue(self, cue_type):
        cue_files = {
            "listening": "listening.mp3",
            "user_done_talking": "user_done_talking.mp3",
            "interrupt_hold": "listening.mp3"
        }
        cue_file = cue_files.get(cue_type)
        if cue_file and os.path.exists(cue_file):
            process = await asyncio.create_subprocess_exec(
                'ffplay', '-autoexit', '-nodisp', cue_file,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await process.communicate()
        else:
            print(f"Audio cue file for {cue_type} not found")

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, callback, manager):
        super().__init__(output_stream)
        self.callback = callback
        self.manager = manager

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript = alt.transcript.strip()
                if result.is_partial:
                    print(f"Partial Transcript: {transcript}")
                    #Interrupt as soon as partial result is detected
                    await self.manager.handle_interruption(transcript)
                else:
                    #print(f"Final Transcript: {transcript}")
                    if transcript:
                        self.callback(transcript)

async def get_transcript(callback, manager):
    try:
        stream = await transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=24000,
            media_encoding="pcm",
        )

        async def send_audio():
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

            loop = asyncio.get_event_loop()
            audio_queue = asyncio.Queue()

            with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype='int16',
                                   channels=1, callback=callback):
                while True:
                    audio_chunk = await audio_queue.get()
                    await stream.input_stream.send_audio_event(audio_chunk=audio_chunk)

        handler = MyEventHandler(stream.output_stream, callback, manager)
        await asyncio.gather(send_audio(), handler.handle_events())

    except Exception as e:
        print(f"Could not transcribe: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""
        self.last_completed_query = ""
        self.silence_start_time = None
        self.query_counter = 1
        self.processing_response = False
        self.interrupted = False
        self.session_id = generate_session_id()
        self.tts = TextToSpeech()
        self.interrupt_flag = asyncio.Event()
        self.last_activity_time = time.time()
        self.speech_in_progress = False
        self.still_talking_timer = None
        self.ignore_list = [
            "uh-huh", "ok", "got it", "sure", "i see", "makes sense", "yeah",
            "mhm", "uh", "oh", "hm", "right", "yep", "yup", "yes", "alright"
        ]
        self.partial_transcript = ""

    def play_greeting(self):
        greeting_file = "greeting.mp3"
        if os.path.exists(greeting_file):
            player_command = ["ffplay", "-autoexit", "-nodisp", greeting_file]
            subprocess.run(player_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Greeting file not found")

    async def handle_interruption(self, transcript, is_final=False):
        self.last_activity_time = time.time()
        if self.tts.playing or self.processing_response:
            cleaned_transcript = re.sub(r'[^\w\s]', '', transcript.lower())
            
            if cleaned_transcript in self.ignore_list:
                print(f"{'Final' if is_final else 'Partial'} Transcript (Ignored): {transcript}")
                return
            
            print(f"{Fore.RED}Interruption detected: {transcript}{Style.RESET_ALL}")
            self.interrupted = True
            self.interrupt_flag.set()
            await self.tts.stop()
            self.processing_response = False
            self.interruption_text = transcript
            print(f"{Fore.RED}Interruption text: {self.interruption_text}{Style.RESET_ALL}")
            await self.tts.play_audio_cue("interrupt_hold")
        else:
            # Not during playback or processing, treat as normal speech
            if is_final:
                self.transcription_response = transcript
                print(f"Final Transcript: {transcript}")
            else:
                self.partial_transcript = transcript
                print(f"Partial Transcript: {transcript}")

    async def process_transcriptions(self):
        while True:
            new_transcription = self.transcription_response or self.partial_transcript
            
            if new_transcription:
                self.last_activity_time = time.time()
                
                # Only reset the still_talking_timer if it's already set and we have new transcription
                if self.still_talking_timer is not None:
                    print(f"Resetting 4-second timer due to new transcription at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    self.still_talking_timer = time.time()  # Reset the timer instead of setting to None

                if self.transcription_response:
                    self.customer_query = self.transcription_response
                    print(f"Current Customer Query: {self.customer_query.strip()}")

                    done_talking_response = self.llm.check_done_talking(self.customer_query)

                    if "not_done_talking" in done_talking_response.lower():
                        if self.still_talking_timer is None:  # Only start the timer if it's not already running
                            print(f"Customer still needs to talk. Starting 4-second timer at {time.strftime('%Y-%m-%d %H:%M:%S')}.")
                            self.still_talking_timer = time.time()
                    else:
                        await self.tts.play_audio_cue("user_done_talking")
                        await self.process_complete_query()

                    self.transcription_response = ""
                
                self.partial_transcript = ""  # Clear partial transcript after processing

            if self.still_talking_timer and (time.time() - self.still_talking_timer) > 4:
                print(f"4 seconds elapsed after 'still talking' at {time.strftime('%Y-%m-%d %H:%M:%S')}. Processing as complete query.")
                await self.tts.play_audio_cue("user_done_talking")
                await self.process_complete_query()
                self.still_talking_timer = None  # Reset the timer after processing

            await asyncio.sleep(0.1)

    async def process_complete_query(self):
        self.still_talking_timer = None
        print(f"Processing complete query: {self.customer_query.strip()}")

        query_id = self.query_counter
        self.query_counter += 1
        print(f"Query ID: {query_id}")

        self.processing_response = True
        print("Processing response started")
        self.interrupt_flag.clear()
        main_responses = await self.llm.get_main_response(self.customer_query.strip(), self.session_id, query_id, self.interrupt_flag)

        for response in main_responses:
            if self.interrupted:
                print("Customer interrupted, discarding current response.")
                break

            print(f"LLM Response (Query ID {response['query_id']}): {response['content']}")
            await self.tts.speak(response['content'])

        if not self.interrupted:
            await self.tts.play_audio_cue("listening")

        self.processing_response = False
        print("Processing response stopped")
        self.customer_query = ""
        self.interrupted = False

    async def handle_idle_timeout(self):
        idle_start_time = None
        first_warning_given = False

        while True:
            current_time = time.time()
            is_active = (
                self.tts.playing or 
                self.processing_response or 
                self.transcription_response or
                (current_time - self.last_activity_time) < 2 or
                self.still_talking_timer is not None  # Add this condition
            )
            
            if not is_active:
                if idle_start_time is None:
                    idle_start_time = time.time()
                    print(f"Idle timeout timer started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                elif time.time() - idle_start_time > 10 and not first_warning_given:
                    print(f"10-second idle threshold reached at {time.strftime('%Y-%m-%d %H:%M:%S')}. Giving first warning.")
                    await self.tts.speak_timeout_message("Hey, are you still there?")
                    first_warning_given = True
                elif time.time() - idle_start_time > 20:
                    print(f"20-second idle threshold reached at {time.strftime('%Y-%m-%d %H:%M:%S')}. Terminating the call.")
                    await self.tts.speak_timeout_message("Thank you for calling, I am terminating the call due to user inactivity.")
                    print("Terminating due to inactivity")
                    os._exit(0)
            else:
                if idle_start_time is not None:
                    print(f"Activity detected. Resetting idle timer at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                idle_start_time = None
                first_warning_given = False

            await asyncio.sleep(1)

    def handle_full_sentence(self, full_sentence):
        if self.tts.playing or self.processing_response:
            asyncio.create_task(self.handle_interruption(full_sentence, is_final=True))
        else:
            self.transcription_response = full_sentence
            print(f"Final Transcript: {full_sentence}")
        
        self.last_activity_time = time.time()
        self.speech_in_progress = True

    async def main(self):
        print(f"Session ID: {self.session_id}")
        self.play_greeting()

        await asyncio.gather(
            get_transcript(self.handle_full_sentence, self),
            self.process_transcriptions(),
            self.handle_idle_timeout()
        )

if __name__ == "__main__":
    manager = ConversationManager()
    try:
        asyncio.run(manager.main())
    except KeyboardInterrupt:
        print("Conversation Manager stopped manually")
