import ollama

response = ollama.chat(model="llama3.2:3b", messages=[
    {"role": "user", "content": "O que é uma função em Python?"}
])

print(response["message"]["content"])