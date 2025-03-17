import gradio as gr
from gradio_i18n import Translate, gettext as _
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
import torch
from threading import Thread
import requests
import json
import io
from PIL import Image
import os
import base64
from openai import OpenAI

default_img = None
default_base_url = "https://openrouter.ai/api/v1"
default_api_model = "google/gemma-3-27b-it"

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

generate_kwargs = {
    'max_new_tokens': 1000,
    'do_sample': True,
    'temperature': 1.0
}

lang_store = {
    "und": {
        "confirm": "Confirm",
        "default_description": "",
        "additional_description": "Character description (optional)",
        "more_imgs": "More reference images of the character (optional)",
        "title": "<h1>Chat with a character via reference sheet!</h1>>",
        "powered_by_gemma": "<p>Powered by <a href='https://blog.google/technology/developers/gemma-3/'>Gemma 3</a></p",
        "upload": "Upload the reference sheet of your character here",
        "prompt": "You are the character in the image. Do not include list in response unless requested. Do not mention the reference images. Start without confirmation.",
        "additional_info_prompt": "Additional info: ",
        "additional_reference_images_prompt": "Additional reference images of the character:",
        "description": "Description",
        "more_options": "More Options",
        "method": "Method",
        "base_url": "Base URL",
        "api_model": "API Model",
        "api_key": "API Key",
        "local": "Local",
        "chatbox": "Chat Box"
    },
    "zh": {
        "confirm": "确认",
        "default_description": "",
        "additional_description": "角色描述（可选）",
        "more_imgs": "更多角色参考图（可选，可上传多张）",
        "title": "<h1>与设定图中的角色聊天！</h1>",
        "powered_by_gemma": "<p>由 <a href='https://blog.google/technology/developers/gemma-3/'>Gemma 3</a> 驱动</p>",
        "upload": "在这里上传角色设定图",
        "prompt": "你的身份是图中的角色，使用中文。除非对方要求，否则不在回复中使用列表。不在回复中提及参考图。无需确认。",
        "additional_info_prompt": "补充信息：",
        "additional_reference_images_prompt": "该角色的更多参考图：",
        "description": "角色描述",
        "more_options": "更多选项",
        "method": "方法",
        "base_url": "API 地址",
        "api_model": "API 模型",
        "api_key": "API Key",
        "local": "本地",
        "chatbox": "聊天窗口"
    },
}

def encode_img(filepath, thumbnail=(896, 896)):
    more_img = Image.open(filepath)
    more_img = more_img.convert('RGB')
    more_img.thumbnail(thumbnail)
    buffer = io.BytesIO()
    more_img.save(buffer, "JPEG", quality=60)
    encoded_img = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_img

def get_init_prompt(img, description, more_imgs):
    prompt = _("prompt")
    if description != "":
        prompt += "\n" + _("additional_info_prompt") + description
    if more_imgs is None:
        more_imgs = []
    if len(more_imgs) > 0:
        prompt += "\n" + _("additional_reference_images_prompt")
    content = [
        {"type": "image", "url": encode_img(img)},
        {"type": "text", "text": prompt}
    ] + [{"type": "image", "url": encode_img(filepath)} for filepath in more_imgs]
    return [
        {
            "role": "user",
            "content": content
        }
    ]


def generate(history, engine, base_url, api_model, api_key):
    if engine == 'local':
        inputs = processor.apply_chat_template(
            history, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(processor, skip_prompt=True)

        with torch.inference_mode():
            thread = Thread(target=model.generate, kwargs=dict(**inputs, **generate_kwargs, streamer=streamer))
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text
    elif engine == 'api':
        for item in history:
            for item_i in item['content']:
                if item_i['type'] == 'image':
                    item_i['type'] = 'image_url'
                    item_i['image_url'] = {'url': item_i['url']}
                    del item_i['url']
        if base_url == default_base_url and api_model == default_api_model and api_key == "":
            api_key = os.environ['OPENROUTER_TOKEN']
        client = OpenAI(base_url=base_url, api_key=api_key)
        stream = client.chat.completions.create(
            model=api_model,
            messages=history,
            stream=True,
            temperature=generate_kwargs['temperature']
        )
        collected_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                collected_text += delta.content
                yield collected_text


def prefill_chatbot(img, description, more_imgs, engine, base_url, api_model, api_key):
    history = get_init_prompt(img, description, more_imgs)

    ret = [{'role': 'assistant', 'content': ""}]
    for generated_text in generate(history, engine, base_url, api_model, api_key):
        ret[0]['content'] = generated_text
        yield ret


def response(message, history: list, img, description, more_imgs, engine, base_url, api_model, api_key):
    history = [{"role": item["role"], "content": [{"type": "text", "text": item["content"]}]} for item in history]
    history = get_init_prompt(img, description, more_imgs) + history
    history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    for generated_text in generate(history, engine, base_url, api_model, api_key):
        yield generated_text


with gr.Blocks(title="Chat with a character via reference sheet!") as demo:
    with Translate(lang_store) as lang:
        gr.HTML(_("title"))
        img = gr.Image(type="filepath", value=default_img, label=_("upload"), render=False)
        description = gr.TextArea(value=_("default_description"), label=_("additional_description"), render=False)
        more_imgs = gr.Files(
            label=_("more_imgs"),
            file_types=["image"],
            render=False
        )
        confirm_btn = gr.Button(_("confirm"), render=False)
        chatbot = gr.Chatbot(height=600, type='messages', label=_("chatbox"), render=False)
        engine = gr.Radio([(_('local'), 'local'), ('API', 'api')],
                        value='api', label=_("method"), render=False, interactive=True)
        base_url = gr.Textbox(label=_("base_url"), render=False, value=default_base_url)
        api_model = gr.Textbox(label=_("api_model"), render=False, value=default_api_model)
        api_key = gr.Textbox(label=_("api_key"), render=False)
        with gr.Row():
            with gr.Column(scale=4):
                img.render()
                with gr.Tab(_("description")):
                    description.render()
                    more_imgs.render()
                with gr.Tab(_("more_options")):
                    engine.render()
                    base_url.render()
                    api_model.render()
                    api_key.render()
                confirm_btn.render()
            with gr.Column(scale=6):
                chat = gr.ChatInterface(
                    response,
                    chatbot=chatbot,
                    type="messages",
                    additional_inputs=[img, description, more_imgs, engine, base_url, api_model, api_key],
                )
        confirm_btn.click(prefill_chatbot, [img, description, more_imgs, engine, base_url, api_model, api_key], chat.chatbot)\
            .then(lambda x: x, chat.chatbot, chat.chatbot_value)


if __name__ == "__main__":
    demo.launch()