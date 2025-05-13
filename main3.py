import os
import re
import json
import base64

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

COMIC_DIR = os.getenv('COMIC_DIR')
MODEL = os.getenv('MODEL')
OPENAI_KEY = os.getenv('OPENAI_KEY')
PROMPT_PATH = os.getenv('PROMPT3_PATH')


def read_layout(layout_path, image_path):
    with open(layout_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        if item.get('path') == image_path:
            return item.get('layout', {})
    return {}


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def get_response(prompt_content, base64_image):
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt_content,
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}'
                        },
                    },
                ]
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content


def run():
    layout_path = os.path.join(COMIC_DIR, 'layout.json')
    for root, dirs, files in os.walk(COMIC_DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                base64_image = image_to_base64(image_path)
                layout = read_layout(layout_path, image_path[len(COMIC_DIR)+1:].replace('\\', '/'))
                layout_json = json.dumps(layout, indent=4)
                prompt_content = read_prompt(PROMPT_PATH)
                # TODO: 增加前情回顾
                prompt_content = prompt_content + '```json\n' + layout_json + '\n```'
                response = get_response(prompt_content, base64_image)
                print('Response:', response)


if __name__ == '__main__':
    run()
