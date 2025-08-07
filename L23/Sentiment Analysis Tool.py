import requests

# Hugging Face API URL for sentiment analysis
api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased"

# Replace with your Hugging Face API token
headers = {
"Authorization": "Bearer YOUR_API_KEY_HERE"
}

# Sample text for sentiment analysis
text = "I love this movie! It was fantastic."

# Send POST request to the Hugging Face API
response = requests.post(api_url, headers=headers, json={"inputs": text})
if response.status_code= 200:
    # Parse the response JSON
    result = response.json()
    print(f"Sentiment: {result[0]['label']} with confidence score: {result[0]['score']}")
else:
    print(f"Error: {response.status_code}")