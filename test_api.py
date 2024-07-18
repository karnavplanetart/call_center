import os
import openai
from dotenv import load_dotenv

load_dotenv()

def test_openai_api():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API key not found.")
        return
    
    openai.api_key = openai_api_key
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        print("API Test Response:", response['choices'][0]['message']['content'].strip())
    except Exception as e:
        print("Error during API call:", str(e))

if __name__ == "__main__":
    test_openai_api()
