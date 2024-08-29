import json

import requests


class OllamaLM:
    def __init__(self) -> None:
        self.chat_url = "http://localhost:11434/api/chat"

    def forward(self, messages):
        res = requests.post(
            self.chat_url,
            json={
                "messages": [item for item in messages],
                "model": "phi3:3.8b-mini-4k-instruct-q6_K",
                "stream": True,
            },
            stream=True,
        )

        res.raise_for_status()

        for line in res.iter_lines():
            yield json.loads(line)["message"]
