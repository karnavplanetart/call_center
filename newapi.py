import requests
import json

send_data = {
    "messages": [{"role": "user", "content": "where is my order ? order number: FP115259217206, email stu0512@avqtest.test", "ts": "2024-7-16 8:23:0", "original_list": [], "refts": ""}],
    "brand": "fp",
    "user_id": "",
    "session_id": "fp-1-c5aa43cf-db5e-4748-882a-65eb012fb10e-fpus",
    "token": "",
    "is_customer": 0,
    "hash": "a1b80e201721118180",
    "site_id": "fpus",
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
                json_object = json.loads(line.decode('utf-8'))
                print(line)

# Example usage
stream_json_request(url)