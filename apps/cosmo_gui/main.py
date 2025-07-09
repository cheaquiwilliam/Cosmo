import os
from openai import OpenAI
import tkinter as tk
from tkinter import scrolledtext
from dotenv import load_dotenv
import tiktoken

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Adjust these as needed
MAX_COMPLETION_TOKENS = 3000
MAX_CONTEXT_TOKENS = 120000  # adjust for your model  # Max tokens for messages history (leave room for response)
MODEL_NAME = "gpt-4"  # or "gpt-4-turbo"


# Initialize tokenizer for the model
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)


def read_persona(persona_name: str, personas_dir="Assets") -> str:
    filepath = os.path.join(personas_dir, f"{persona_name}")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Persona file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def count_message_tokens(messages):
    """
    Counts the total tokens for the messages list, including overhead.
    """
    total_tokens = 0
    for message in messages:
        # Every message has ~4 tokens overhead (OpenAI docs)
        total_tokens += 4
        for key, value in message.items():
            total_tokens += len(tokenizer.encode(value))
    total_tokens += 2  # Priming tokens (OpenAI docs)
    return total_tokens


def trim_history(messages):
    """
    Trim oldest messages while keeping under MAX_TOKENS,
    always preserving the first two (system + persona).
    """
    system_and_persona = messages[:2]
    rest = messages[2:]
    
    while rest and (count_message_tokens(system_and_persona + rest) + MAX_COMPLETION_TOKENS > MAX_CONTEXT_TOKENS):
        rest.pop(0)
    
    return system_and_persona + rest


def query_openai(messages, model=MODEL_NAME, temperature=0.7, max_tokens=MAX_COMPLETION_TOKENS):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


class ChatApp:
    def __init__(self, root, persona_text, system_prompt):
        self.root = root
        self.root.title(f"{persona_text} Chat")

        self.messages = [
            {"role": "system", "content": system_prompt}
            # {"role": "user", "content": persona_text},
        ]

        # Chat display
        self.chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap='word')
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # User input field
        self.user_input = tk.Entry(root, width=80)
        self.user_input.pack(padx=10, pady=(0,10), fill=tk.X)
        self.user_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(padx=10, pady=(0,10))

        # self.display_message("System", system_prompt)
        self.display_message(f"Hi this is {persona_text.title()}", " How can I assist you today?")

    def display_message(self, sender, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)

    def trim_history(self):
        self.messages = trim_history(self.messages)

    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.user_input.delete(0, tk.END)

        if user_text.lower() in ['exit', 'quit']:
            self.root.quit()
            return

        self.display_message("You", user_text)
        self.messages.append({"role": "user", "content": user_text})

        self.trim_history()

        self.root.config(cursor="wait")
        self.root.update()

        try:
            response = query_openai(self.messages)
        except Exception as e:
            response = f"Error: {e}"

        self.messages.append({"role": "assistant", "content": response})
        self.display_message("Assistant", response)

        self.root.config(cursor="")


def main():
    persona_name = input("Enter gpt name (e.g., cue or jared): ").strip().lower()
    try:
        gpt = read_persona(persona_name)
    except FileNotFoundError as e:
        print(e)
        return

    system_prompt = gpt

    root = tk.Tk()
    app = ChatApp(root, persona_name, system_prompt)
    root.mainloop()


if __name__ == "__main__":
    main()
