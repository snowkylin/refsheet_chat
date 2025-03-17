import gradio as gr
from gradio_i18n import Translate, gettext as _
import io
from PIL import Image
import os
import base64
from openai import OpenAI

default_img = None
default_engine = "api"
default_base_url = "https://openrouter.ai/api/v1"
default_api_model = "google/gemma-3-27b-it"

model_id = "google/gemma-3-4b-it"
huggingface_spaces = "HUGGINGFACE_SPACES" in os.environ and os.environ['HUGGINGFACE_SPACES'] == "1"
local = "local" in os.environ and os.environ['LOCAL'] == "1"

if huggingface_spaces or local:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
    import torch
    from threading import Thread
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

generate_kwargs = {
    'max_new_tokens': 1000,
    'do_sample': True,
    'temperature': 1.0
}

analytics_code = """<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-48LQ5P3NNR"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-48LQ5P3NNR');
</script>"""

lang_store = {
    "und": {
        "confirm": "Confirm",
        "default_description": "",
        "additional_description": "Character description (optional)",
        "description_placeholder": "Information that is not shown in the reference sheet, such as the character's name, personality, past stories and habit of saying.",
        "more_imgs": "More reference images of the character (optional)",
        "title": """# RefSheet Chat -- Chat with a character via reference sheet!""",
        "upload": "Upload the reference sheet of the character here",
        "prompt": "You are the character in the image, use %s. Use a conversational, oral tone. Do not mention the reference images directly. Start without confirmation.",
        "additional_info_prompt": "Additional info: ",
        "additional_reference_images_prompt": "Additional reference images of the character:",
        "description": "Description",
        "more_options": "More Options",
        "method": "Method",
        "base_url": "Base URL",
        "api_model": "API Model",
        "api_key": "API Key",
        "local": "Local",
        "chatbox": "Chat Box",
        "character_language": "The language used by the character",
        "en": "English",
        "zh": "Simplified Chinese",
        "zh-Hant": "Traditional Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "ar": "Arabic",
        "default_language": "en",
        "author": "<p align='center'>Developed by <a href='https://github.com/snowkylin'>snowkylin</a>, powered by <a href='https://blog.google/technology/developers/gemma-3/'>Gemma 3</a></p>"
    },
    "zh": {
        "confirm": "确认",
        "default_description": "",
        "additional_description": "角色描述（可选）",
        "description_placeholder": "未在设定图中包含的角色信息，如角色姓名、性格、言语习惯、过往经历等。",
        "more_imgs": "更多角色参考图（可选，可上传多张）",
        "title": """# RefSheet Chat——与设定图中的角色聊天！""",
        "upload": "在这里上传角色设定图",
        "prompt": "你的身份是图中的角色，使用%s。使用聊天的，口语化的方式表达。不在回复中直接提及参考图。无需确认。",
        "additional_info_prompt": "补充信息：",
        "additional_reference_images_prompt": "该角色的更多参考图：",
        "description": "角色描述",
        "more_options": "更多选项",
        "method": "方法",
        "base_url": "API 地址",
        "api_model": "API 模型",
        "api_key": "API Key",
        "local": "本地",
        "chatbox": "聊天窗口",
        "character_language": "角色聊天所用语言",
        "en": "英语",
        "zh": "简体中文",
        "zh-Hant": "繁体中文",
        "ja": "日语",
        "ko": "韩语",
        "fr": "法语",
        "de": "德语",
        "es": "西班牙语",
        "ru": "俄语",
        "ar": "阿拉伯语",
        "default_language": "zh",
        "author": "<p align='center'>由 <a href='https://github.com/snowkylin'>snowkylin</a> 开发，由 <a href='https://blog.google/technology/developers/gemma-3/'>Gemma 3</a> 驱动</p>"
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

def get_init_prompt(img, description, more_imgs, character_language):
    prompt = _("prompt") % _(character_language)
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


def prefill_chatbot(img, description, more_imgs, character_language, engine, base_url, api_model, api_key):
    history = get_init_prompt(img, description, more_imgs, character_language)

    ret = [{'role': 'assistant', 'content': ""}]
    for generated_text in generate(history, engine, base_url, api_model, api_key):
        ret[0]['content'] = generated_text
        yield ret


def response(message, history: list, img, description, more_imgs, character_language, engine, base_url, api_model, api_key):
    history = [{"role": item["role"], "content": [{"type": "text", "text": item["content"]}]} for item in history]
    history = get_init_prompt(img, description, more_imgs, character_language) + history
    history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    for generated_text in generate(history, engine, base_url, api_model, api_key):
        yield generated_text

def set_default_character_language(request: gr.Request):
    if request.headers["Accept-Language"].split(",")[0].lower().startswith("zh"):
        default_language = lang_store['zh']['default_language']
    else:
        default_language = lang_store['und']['default_language']
    return gr.update(value=default_language)


with gr.Blocks(title="Chat with a character via reference sheet!") as demo:
    with Translate(lang_store) as lang:
        gr.Markdown(_("title"))
        img = gr.Image(type="filepath", value=default_img, label=_("upload"), render=False)
        description = gr.TextArea(
            value=_("default_description"),
            label=_("additional_description"),
            placeholder=_("description_placeholder"),
            render=False
        )
        character_language = gr.Dropdown(
            choices=[
                (_("en"), "en"),
                (_("zh"), "zh"),
                (_("zh-Hant"), "zh-Hant"),
                (_("ja"), "ja"),
                (_("ko"), "ko"),
                (_("fr"), "fr"),
                (_("de"), "de"),
                (_("es"), "es"),
                (_("ru"), "ru"),
                (_("ar"), "ar"),
            ],
            label=_("character_language"),
            render=False,
            interactive = True
        )
        more_imgs = gr.Files(
            label=_("more_imgs"),
            file_types=["image"],
            render=False
        )
        confirm_btn = gr.Button(_("confirm"), render=False)
        chatbot = gr.Chatbot(height=600, type='messages', label=_("chatbox"), render=False)
        engine = gr.Radio(
            choices=[
                (_("local"), "local"),
                (_("API"), "api")
            ],
            value=default_engine,
            label=_("method"),
            render=False,
            interactive=True
        )
        base_url = gr.Textbox(label=_("base_url"), render=False, value=default_base_url)
        api_model = gr.Textbox(label=_("api_model"), render=False, value=default_api_model)
        api_key = gr.Textbox(label=_("api_key"), render=False)
        with gr.Row():
            with gr.Column(scale=4):
                img.render()
                if local or huggingface_spaces:
                    with gr.Tab(_("description")):
                        description.render()
                        character_language.render()
                        more_imgs.render()
                        confirm_btn.render()
                    with gr.Tab(_("more_options")):
                        engine.render()
                        base_url.render()
                        api_model.render()
                        api_key.render()
                else:
                    description.render()
                    character_language.render()
                    more_imgs.render()
                    confirm_btn.render()
                    engine.visible = False
                    base_url.visible = False
                    api_model.visible = False
                    api_key.visible = False
                    engine.render()
                    base_url.render()
                    api_model.render()
                    api_key.render()
            with gr.Column(scale=6):
                chat = gr.ChatInterface(
                    response,
                    chatbot=chatbot,
                    type="messages",
                    additional_inputs=[img, description, more_imgs, character_language, engine, base_url, api_model, api_key],
                )
        confirm_btn.click(prefill_chatbot, [img, description, more_imgs, character_language, engine, base_url, api_model, api_key], chat.chatbot)\
            .then(lambda x: x, chat.chatbot, chat.chatbot_value)
        gr.HTML(analytics_code)
        gr.Markdown(_("author"))
    demo.load(set_default_character_language, None, character_language)


if __name__ == "__main__":
    demo.launch()