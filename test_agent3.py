import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import json

from langchain.memory import ConversationBufferMemory
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

# Define the URL and headers for PlanetArt API
url = 'https://ai.planetart.com/api/v1/chat/chat?session_id=sw-1-a94d7dcd-5285-48cd-9f4c-8173b6faffec-sti'
headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://chatbot.planetart.com',
    'priority': 'u=1, i',
    'referer': 'https://chatbot.planetart.com/',
    'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

# Initialize the session
session_id = "sw-1-a94d7dcd-5285-48cd-9f4c-8173b6faffec-sti"
messages = [{"role": "assistant", "content": "Welcome to simplytoimpress.com Customer Care! How can I help you today?", "ts": "2024-07-08 18:27:21"}]

class LanguageModelProcessor:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def send_message_to_api(self, user_message):
        messages.append({"role": "user", "content": user_message, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        payload = {
            "messages": messages,
            "brand": "sw",
            "user_id": "",
            "session_id": session_id,
            "debug": "0",
            "token": "",
            "is_customer": 1,
            "hash": "977094941720463361",
            "site_id": "sti",
            "url_query": "w%3D440px%26h%3D580px%26user_id%3D%26brand%3Dsw%26site_id%3Dsti%26is_customer%3D1%26region%3Dus%26debug%3D0%26newpc%3D0%26crm_customer_url%3Dhttps%253A%252F%252Fwww.simplytoimpress.com%252F%253Futm_source%253DAdWords%2526amp%253Butm_medium%253DPPC%2526amp%253Butm_source%253DAdWords%2526amp%253Butm_medium%253Dcpc%2526amp%253Butm_campaign%253D21555188%2526amp%253Butm_term%253Dsimply%252520to%252520impress%2526amp%253Bgad_source%253D1%26start_page_name%3DSimply%2Bto%2BImpress%2B%257C%2BBirth%2BAnnouncements%252C%2BInvitations%252C%2BHoliday%2BCards%26t%3D477906"
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_data = response.json()
        assistant_messages = [msg['content'] for msg in response_data['messages'] if msg['role'] == 'assistant']
        for assistant_message in assistant_messages:
            print(f"Assistant: {assistant_message}")
            messages.append({"role": "assistant", "content": assistant_message, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        return assistant_messages

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")
        
        print(f"Converting text to speech: {text}")
        
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=default&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        try:
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            start_time = time.time()  # Record the time before sending the request
            first_byte_time = None  # Initialize a variable to store the time when the first byte is received

            with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
                if r.status_code != 200:
                    print(f"Failed to get TTS audio: {r.status_code}, {r.text}")
                    return

                audio_received = False
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        audio_received = True
                        if first_byte_time is None:  # Check if this is the first chunk received
                            first_byte_time = time.time()  # Record the time when the first byte is received
                            ttfb = int((first_byte_time - start_time) * 1000)  # Calculate the time to first byte
                            print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()

                if not audio_received:
                    print("No audio data received from Deepgram TTS API.")
                
            if player_process.stdin:
                player_process.stdin.close()
            player_process.wait()

            stdout, stderr = player_process.communicate()
            if stdout:
                print(f"ffplay stdout: {stdout.decode()}")
            if stderr:
                print(f"ffplay stderr: {stderr.decode()}")
        except Exception as e:
            print(f"Error in TTS processing: {e}")

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
        deepgram: DeepgramClient = DeepgramClient("", config)

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

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check if the customer is done talking
            done_talking_responses = self.llm.send_message_to_api(self.transcription_response)
            
            for done_talking_response in done_talking_responses:
                if "customer done talking" in done_talking_response.lower():
                    self.customer_query += " " + self.transcription_response
                    print(f"Customer Query (Complete): {self.customer_query.strip()}")  # Log the complete customer query
                    # Get the main response from the API
                    main_responses = self.llm.send_message_to_api(self.customer_query.strip())
                    tts = TextToSpeech()
                    for response in main_responses:
                        print(f"Speaking response: {response}")
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


