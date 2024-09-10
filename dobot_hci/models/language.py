import json

import requests


class OllamaLM:
    def __init__(self, model_name: str = "llama3:8b-instruct-q4_1") -> None:
        self.chat_url = "http://localhost:11434/api/chat"
        self.model_name = model_name

    def forward(self, messages):
        res = requests.post(
            self.chat_url,
            json={
                "messages": [item for item in messages],
                "model": self.model_name,
                "stream": True,
            },
            stream=True,
        )

        res.raise_for_status()

        for line in res.iter_lines():
            yield json.loads(line)["message"]
