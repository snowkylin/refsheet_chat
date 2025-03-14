from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "snowkylin.jpg"},
            {"type": "text", "text": "You are the character in the image. Start without confirmation."}
            # {"type": "text", "text": "你的身份是图中的角色，使用中文。无需确认。"}
        ]
    }
]

generate_kwargs = {
    'max_new_tokens': 1000,
    'do_sample': True,
    'temperature': 1.0
}

while True:
    response = pipe(text=messages, generate_kwargs=generate_kwargs)

    messages = response[0]['generated_text']
    print(messages[-1]["content"])

    content = input(">> ")

    messages.append(
        {"role": "user", "content": content}
    )