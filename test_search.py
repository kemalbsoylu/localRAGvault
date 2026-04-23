import requests

url = "http://127.0.0.1:8000/search/"

data = {
    "query": "What database does localRAGvault use?",
    "top_k": 2
}

print(f"Asking: '{data['query']}'\n")

response = requests.post(url, json=data)

if response.status_code == 200:
    results = response.json()["results"]
    for i, res in enumerate(results, 1):
        print(f"--- Match {i} (Similarity: {res['similarity']}) ---")
        print(f"File: {res['filename']}")
        print(f"Content: {res['content']}\n")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
