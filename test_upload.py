import requests
import os

url = "http://127.0.0.1:8000/upload/"
filename = "sample_test.txt"

# 1. Create a dummy file with some text to chunk
dummy_text = """
localRAGvault is a privacy-first, fully local RAG architecture.
It uses Ollama to serve generation and embedding models directly on the host machine.
By utilizing PostgreSQL and pgvector, it stores document embeddings securely without relying on external APIs or cloud services.
The backend is powered by Python and FastAPI, ensuring high performance and asynchronous request handling.
"""

with open(filename, "w", encoding="utf-8") as f:
    f.write(dummy_text)

# 2. Send the file to our FastAPI endpoint
print(f"Uploading {filename} to {url}...")
with open(filename, "rb") as f:
    files = {"file": (filename, f, "text/plain")}
    response = requests.post(url, files=files)

# 3. Print the results
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(response.json())

# Clean up the dummy file
os.remove(filename)
