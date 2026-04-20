import requests
import json
import sys

url = "http://localhost:8000/api/v1/search"
payload = {
    "messages": [
        {"role": "user", "content": "24-year-old male with sudden vision loss and severe headache"}
    ]
}

try:
    print("Initiating request to Clinsight Backend...")
    response = requests.post(url, json=payload, stream=True)
    if response.status_code != 200:
        print(f"Server Error {response.status_code}: {response.text}")
        sys.exit(1)
        
    full_text = ""
    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            if chunk.startswith("data: "):
                data = json.loads(chunk[6:])
                if data.get("type") == "chunk":
                    full_text += data.get("content", "")
                    sys.stdout.write(data.get("content", ""))
                    sys.stdout.flush()
                elif data.get("type") == "metadata":
                    print("\n\n--- METADATA ---")
                    print(json.dumps(data.get("content", {}), indent=2))
                elif data.get("type") == "error":
                    print(f"\nERROR: {data.get('content')}")
            
except Exception as e:
    print(f"Failed to connect or parse: {e}")
