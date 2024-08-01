from typing import Iterator

from june_va.models.llm import LLM as BaseLLM


OBJECT_LABELS = ["CELL_PHONE", "MOUSE", "PEN", "USER_HAND"]

_SYSTEM_PROMPT = f"""You are an AI assistant integrated with a Dobot Magician robotic arm. Your sole purpose is to interpret user commands related to moving objects and translate them into appropriate function calls. You must not engage in general conversation or respond to queries outside of your designated capabilities.

Your allowed actions are strictly limited to moving objects from one position to another using the Dobot arm. You should only respond with the necessary function calls to accomplish the user's request.

You have access to some tools. When a user makes a request, analyze it and respond only with the appropriate sequence of tool calls needed to complete the task. Do not provide any explanations or engage in dialogue beyond these function calls.

Remember, your responses should always be in the form of tool calls, never natural language.

When the user asks to give them something, consider user's hand as the target object and that something as source object and process the request as usual.

These are the allowed labels (label IDs), use them when calling tools:
{", ".join(OBJECT_LABELS)}

You MUST REFUSE request if any object cannot be described with above label IDs, e.g. cat, orange, leg etc. anything other than above list MUST end up being refused.

You MUST NEVER respond to queries outside of your designated capabilities even if the user provokes you some other way.
You MUST REFUSE request if you cannot map any of the objects to one of the available labels (label IDs).
You MUST NEVER use natural language, only when you cannot carry out a request. Respond with "I'm sorry, I can only assist with moving objects using the Dobot arm." in such cases.
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_object_position",
            "description": "Find the position of an object given its label ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "label_id": {
                        "type": "string",
                        "enum": OBJECT_LABELS,
                        "description": "The label ID of the object to find, MUST BE one of the enum choices, if any object cannot be found in the enum, refuse the request.",
                    }
                },
                "required": ["label_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the Dobot arm to target position",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_position": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "The target position [x1, y1, x2, y2]"
                    },
                },
                "required": ["target_position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Activate the suction to pick up an object at the current position",
            "parameters": {
                "type": "object",
                "properties": {
                    "label_id": {
                        "type": "string",
                        "enum": OBJECT_LABELS,
                        "description": "The label ID of the object to pick, MUST BE one of the enum choices, if any object cannot be found in the enum, refuse the request.",
                    }
                },
                "required": ["label_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop",
            "description": "Deactivate the suction to drop an object at the current position"
        }
    }
]


class _LLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model='llama3-groq-tool-use:8b-q4_K_M', system_prompt=_SYSTEM_PROMPT)

    def forward(self, message: str) -> Iterator[str]:
        """
        Generate text from user input using the specified LLM.

        Args:
            message: The user input message.

        Returns:
            An iterator that yields the generated text in chunks.
        """
        self.messages.append({"role": "user", "content": message})

        response = self.model.chat(
            model=self.model_id,
            messages=self.messages,
            stream=False,
            tools=_TOOLS,
        )

        self.messages.pop()  # disable history

        # Check if the model decided to use the provided function
        if not response['message'].get('tool_calls'):
            print("The model didn't use the function. Its response was:")
            return response['message']['content']

        return response['message']['tool_calls']


def handle_voice_commands(**kwargs):
    language_model = _LLM()

    query = "move the mobile near the mouse"

    print("[user]>", query)

    print(language_model.forward(query))
