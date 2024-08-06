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
from colorama import init, Fore, Style
import uuid
import aiohttp  # Import aiohttp for asynchronous HTTP requests

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
BACKEND_API_URL = "https://ai.planetart.com/api/v1/chat/chat-stream"
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # Only if using temporary credentials

transcribe_client = TranscribeStreamingClient(region=AWS_REGION)

def generate_session_id():
    # Generate a new UUID
    unique_id = uuid.uuid4()
    return f"sw-1-{unique_id}-sti"

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_history = []

        with open('system_prompt_check.txt', 'r') as file:
            self.system_prompt_check = file.read().strip()

    def get_combined_conversation(self, n):
        conv_combined_text = ""
        # Get the last n messages from conversation history
        last_n_messages = self.conversation_history[-n:]
        
        for message in last_n_messages:
            role = "User" if message["role"] == "user" else "Assistant"
            conv_combined_text += f"{role}: {message['content']}\n"
        
        return conv_combined_text.strip()

    def check_done_talking(self, text, n=5):
        combined_conversation = self.get_combined_conversation(n)
        combined_conversation += f"\nUser: {text}"  # Add the current user message

        # Print the combined conversation text for clarity
        print(f"Combined Conversation for LLM Check:\n{combined_conversation}\n")

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
        
        # Color-coded output
        if "not_done_talking" in response['text'].lower():
            print(f"{Fore.YELLOW}LLM Check ({elapsed_time}ms): {response['text']}{Style.RESET_ALL}")
        elif "customer_done_talking" in response['text'].lower():
            print(f"{Fore.GREEN}LLM Check ({elapsed_time}ms): {response['text']}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}LLM Check ({elapsed_time}ms): {response['text']}{Style.RESET_ALL}")

        return response['text']

    async def get_main_response(self, text, session_id, query_id, interrupt_flag, tts):
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
            "url_query": "is_customer%3D0%26brand%3Dfp%26site_id%3Dfpus%26llm_type%3Dclaude35-sonnet"
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    full_message = ""
                    first_segment_sent = False
                    background_music_task = None

                    async for line in response.content:
                        if line:
                            current_time = time.time()
                            print(f"Received part of the response at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
                            json_object = json.loads(line.decode('utf-8'))
                            print(line)
                            assistant_replies = json_object.get("messages", [])

                            for reply in assistant_replies:
                                if interrupt_flag.is_set():
                                    print("Processing interrupted by customer")
                                    return

                                content = reply.get("content", "")
                                full_message += content

                                if not first_segment_sent and len(full_message.split()) > 20:
                                    words = full_message.split()
                                    first_five_words = ' '.join(words[:5])
                                    rest_of_message = ' '.join(words[5:])

                                    # Find the first period after the first five words
                                    period_index = rest_of_message.find('.')
                                    if period_index != -1:
                                        first_segment = first_five_words + ' ' + rest_of_message[:period_index+1]
                                        second_segment = rest_of_message[period_index+1:].strip()
                                        
                                        # Send first segment to TTS
                                        await tts.speak(first_segment)
                                        
                                        # Start looping background music
                                        background_music_task = asyncio.create_task(tts.play_background_music())
                                        
                                        # Send second segment to TTS (but don't play yet)
                                        await tts.prepare_audio(second_segment)
                                        
                                        # Stop background music and play the second segment
                                        await tts.play_prepared_audio()
                                        
                                        first_segment_sent = True
                                    else:
                                        # If no period found, speak the entire message
                                        await tts.speak(full_message)
                                        first_segment_sent = True

                                assistant_message = {
                                    "role": "assistant",
                                    "content": content,
                                    "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
                                    "query_id": query_id
                                }
                                self.conversation_history.append(assistant_message)

                    if not first_segment_sent:
                        # If the message was short, just play it all at once
                        await tts.speak(full_message)

        except Exception as e:
            print(f"Error during get_main_response: {e}")
            await tts.speak("There was an error processing your request. Please try again later.")

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, callback, manager):
        super().__init__(output_stream)
        self.callback = callback
        self.manager = manager

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                if not result.is_partial:
                    full_sentence = alt.transcript.strip()
                    if full_sentence:
                        print(f"Human: {full_sentence}")
                        self.callback(full_sentence)
                else:
                    self.manager.last_activity_time = time.time()  # Update last activity time for partial results
                    if self.manager.processing_response:
                        print("Customer interrupted during processing response")
                    if self.manager.tts.playing:
                        print("Customer interrupted during TTS playback")

async def get_transcript(callback, manager):
    try:
        stream = await transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
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
    
class TextToSpeech:
    def __init__(self):
        self.polly = boto3.client('polly', region_name=AWS_REGION)
        self.playing = False  # Flag to indicate if TTS is playing
        self.process = None  # Store the process handle
        self.prepared_audio = None
        self.background_music_process = None
        
    async def play_background_music(self):
            if self.background_music_process is None:
                self.background_music_process = await asyncio.create_subprocess_exec(
                    'ffplay', '-nodisp', '-loop', '-1', 'rhythmic_robot_typing.mp3',
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

    async def stop_background_music(self):
            if self.background_music_process:
                self.background_music_process.terminate()
                await self.background_music_process.wait()
                self.background_music_process = None
            
    def format_ssml(self, text):
        # Regular expression patterns
        email_pattern = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
        number_pattern = re.compile(r'\b\d{5,}\b')
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        def replace_email_with_ssml(match):
            local_part = match.group(1)
            domain_part = match.group(2)
            return f"<break time='300ms'/><say-as interpret-as='characters'>{local_part}</say-as><break time='300ms'/>@{domain_part}"

        def replace_number_with_ssml(match):
            number = match.group(0)
            grouped_number = ' '.join([number[i:i+2] for i in range(0, len(number), 2)])
            return f"<say-as interpret-as='telephone'>{grouped_number}</say-as>"

        # Remove URLs
        text = url_pattern.sub('', text)

        ssml_text = email_pattern.sub(replace_email_with_ssml, text)
        ssml_text = number_pattern.sub(replace_number_with_ssml, ssml_text)
        return f"<speak>{ssml_text}</speak>"

    async def prepare_audio(self, text):
        ssml_text = self.format_ssml(text)
        response = self.polly.synthesize_speech(
            Text=ssml_text,
            TextType='ssml',
            Engine="generative",
            OutputFormat='mp3',
            VoiceId='Ruth'
        )
        if 'AudioStream' in response:
            self.prepared_audio = response['AudioStream'].read()

    async def play_prepared_audio(self):
        await self.stop_background_music()  # Stop background music before playing prepared audio
        if self.prepared_audio:
            with open('speech.mp3', 'wb') as file:
                file.write(self.prepared_audio)
            
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
            self.prepared_audio = None

    async def speak(self, text):
        await self.stop_background_music()  # Stop background music before speaking
        def format_ssml(text):
            # Regular expression patterns
            email_pattern = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
            number_pattern = re.compile(r'\b\d{5,}\b')
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

            def replace_email_with_ssml(match):
                local_part = match.group(1)
                domain_part = match.group(2)
                return f"<break time='300ms'/><say-as interpret-as='characters'>{local_part}</say-as><break time='300ms'/>@{domain_part}"

            def replace_number_with_ssml(match):
                number = match.group(0)
                grouped_number = ' '.join([number[i:i+2] for i in range(0, len(number), 2)])
                return f"<say-as interpret-as='telephone'>{grouped_number}</say-as>"

            # Remove URLs
            text = url_pattern.sub('', text)

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
            self.process.terminate()  # Terminate the TTS playback
            await self.process.wait()
            self.playing = False
            print("TTS playback stopped by interruption")

    async def speak_timeout_message(self, message):
        await self.speak(message)

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.customer_query = ""
        self.last_completed_query = ""  # Store the last completed query
        self.silence_start_time = None
        self.query_counter = 1  # Initialize the query counter
        self.processing_response = False  # Flag to indicate if a response is being processed
        self.interrupted = False  # Flag to indicate if the conversation was interrupted
        self.session_id = generate_session_id()
        self.tts = TextToSpeech()  # Initialize TTS
        self.interrupt_flag = asyncio.Event()  # Interrupt flag
        self.last_activity_time = time.time()
        self.speech_in_progress = False
        self.still_talking_timer = None

    def play_greeting(self):
        # Play the pre-recorded greeting message
        greeting_file = "greeting.mp3"  # Path to your pre-recorded greeting audio file
        if os.path.exists(greeting_file):
            player_command = ["ffplay", "-autoexit", "-nodisp", greeting_file]
            subprocess.run(player_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Greeting file not found")

    async def process_transcriptions(self):
        while True:
            if self.transcription_response:
                self.last_activity_time = time.time()
                if self.processing_response:
                    # If already processing a response, mark as interrupted and combine inputs
                    self.interrupted = True
                    self.customer_query += " " + self.transcription_response
                    print(f"Customer Query (Interrupted and Appending): {self.customer_query.strip()}")
                    self.transcription_response = ""  # Reset transcription_response
                    self.interrupt_flag.set()  # Set the interrupt flag
                    continue

                self.customer_query += " " + self.transcription_response
                self.transcription_response = ""  # Reset transcription_response after appending
                print(f"Current Customer Query: {self.customer_query.strip()}")

                done_talking_response = self.llm.check_done_talking(self.customer_query)

                if "not_done_talking" in done_talking_response.lower():
                    print(f"{Fore.YELLOW}Customer still needs to talk. Starting/resetting 20-second timer.{Style.RESET_ALL}")
                    self.still_talking_timer = time.time()
                else:
                    print(f"{Fore.GREEN}Customer done talking. Processing complete query.{Style.RESET_ALL}")
                    await self.process_complete_query()

            # Check if the 5-second timer for "still talking" has elapsed
            if self.still_talking_timer and (time.time() - self.still_talking_timer) > 20:
                print("20 seconds elapsed after 'still talking'. Processing as complete query.")
                await self.process_complete_query()

            await asyncio.sleep(0.1)  # Check frequently

    async def process_complete_query(self):
        self.still_talking_timer = None  # Reset the timer
        print(f"Processing complete query: {self.customer_query.strip()}")

        query_id = self.query_counter
        self.query_counter += 1
        print(f"Query ID: {query_id}")

        self.processing_response = True
        print("Processing response started")
        self.interrupt_flag.clear()
        await self.llm.get_main_response(self.customer_query.strip(), self.session_id, query_id, self.interrupt_flag, self.tts)

        self.processing_response = False
        print("Processing response stopped")
        self.customer_query = ""  # Reset customer_query after getting the main response

    async def handle_idle_timeout(self):
        idle_start_time = None
        first_warning_given = False

        while True:
            current_time = time.time()
            is_active = (
                self.tts.playing or 
                self.processing_response or 
                self.transcription_response or
                (current_time - self.last_activity_time) < 2
            )
            
            if not is_active and not self.tts.playing and not self.processing_response:
                if idle_start_time is None:
                    idle_start_time = time.time()
                    print(f"Idle timeout timer started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                elif time.time() - idle_start_time > 10 and not first_warning_given:
                    print(f"10-second idle threshold reached at {time.strftime('%Y-%m-%d %H:%M:%S')} . Giving first warning.")
                    await self.tts.speak_timeout_message("Hey, are you still there?")
                    first_warning_given = True
                    idle_start_time = time.time()  # Reset the timer after warning
                elif time.time() - idle_start_time > 20:
                    print(f"20-second idle threshold reached at {time.strftime('%Y-%m-%d %H:%M:%S')}. Terminating the call.")
                    await self.tts.speak_timeout_message("Thank you for calling, I am terminating the call due to user inactivity.")
                    print("Terminating due to inactivity")
                    os._exit(0)  # Force exit the program
            else:
                if idle_start_time is not None:
                    print(f"Activity detected. Resetting idle timer at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                idle_start_time = None
                first_warning_given = False

            await asyncio.sleep(1)  # Check every second

    def handle_full_sentence(self, full_sentence):
        self.transcription_response = full_sentence
        self.last_activity_time = time.time()
        self.speech_in_progress = True
        
        if self.processing_response:
            self.interrupt_flag.set()
        if self.tts.playing:
            asyncio.create_task(self.tts.stop())

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