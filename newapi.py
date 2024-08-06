import requests
import json
import time

send_data = {
    "messages": [{"role": "user", "content": "do you have any books about landing on the moon", "ts": "2024-7-16 8:23:0", "original_list": [], "refts": ""}],
    "brand": "ism",
    "user_id": "",
    "session_id": "fp-1-c5aa43cf-db5e-4748-882a-65eb012fb10e-fpus",
    "token": "",
    "is_customer": 0,
    "hash": "a1b80e201721118180",
    "site_id": "ismus",
    "url_query": "is_customer%3D0%26brand%3Dfp%26site_id%3Dfpus%26llm_type%3Dclaude35-sonnet"
}

# url = "https://chatbot-stream-script.fppre-us.planetart.com/api/v1/chat/chat-stream"
# url = "http://127.0.0.1:5001/api/v1/chat/chat-stream"
url = "https://ai.planetart.com/api/v1/chat/chat-stream"

def stream_json_request(url):
    with requests.post(url, json=send_data, stream=True) as response:
        response.raise_for_status()  # Raise an exception for HTTP errors
        for line in response.iter_lines():
            if line:
                current_time = time.time()
                print(f"Received part of the response at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
                json_object = json.loads(line.decode('utf-8'))
                print(line)

# Example usage
stream_json_request(url)