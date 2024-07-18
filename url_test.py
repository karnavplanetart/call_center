import requests
import json
import time

# Define the URL and headers
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

# Function to send a message and get the response
def send_message(user_message):
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
    assistant_message = response_data['messages'][-1]['content']
    print(f"Assistant: {assistant_message}")
    messages.append({"role": "assistant", "content": assistant_message, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})

# Start the chat
print("Assistant: Welcome to simplytoimpress.com Customer Care! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chat ended.")
        break
    send_message(user_input)