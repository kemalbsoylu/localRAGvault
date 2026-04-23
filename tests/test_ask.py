import requests

url = "http://127.0.0.1:8000/ask/"

data = {
    "query": "What is localRAGvault's privacy approach?",
    "top_k": 2
}

print(f"Thinking about: '{data['query']}'...\n")

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("* gemma4 Answer:")
    print(result["answer"])
    print("\n* Sources Used:")
    for source in result["sources"]:
        print(f"- {source['filename']} (Relevance: {source['similarity']})")
else:
    print(f"Error: {response.status_code} - {response.text}")
