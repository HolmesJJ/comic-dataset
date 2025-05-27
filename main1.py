import os
import re
import cv2
import json
import base64

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

COMIC_DIR = os.getenv('COMIC_DIR')
MODEL = os.getenv('GPT_MODEL')  # GPT_MODEL, QWEN_MODEL, CLAUDE_MODEL, GEMINI_MODEL
MODEL_KEY = os.getenv('GPT_KEY')  # GPT_KEY, QWEN_KEY, CLAUDE_KEY, GEMINI_KEY
MODEL_URL = os.getenv('QWEN_URL')  # QWEN_URL, CLAUDE_URL, GEMINI_URL
PROMPT_PATH = os.getenv('PROMPT1_PATH')


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def extract_json(text):
    try:
        json_data = json.loads(text)
        if json_data != {}:
            return json_data
    except (Exception,):
        pass
    json_match = re.search(r'```(?:json)?(.+?)```', text, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        try:
            json_data = json.loads(json_content)
            if json_data != {}:
                return json_data
        except (Exception,):
            pass
    dict_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if dict_match:
        dict_content = dict_match.group(1).strip()
        try:
            dict_data = eval(dict_content)
            if isinstance(dict_data, dict) and dict_data != {}:
                return dict_data
        except (Exception,):
            pass
    return {}


def draw_bounding_boxes(image_path, response_json):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    for obj, bbox in response_json.items():
        x, y, w, h = bbox
        start_point = (int(x * width), int(y * height))
        end_point = (int((x + w) * width), int((y + h) * height))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        label_position = (start_point[0], start_point[1] - 10 if start_point[1] - 10 > 10 else start_point[1] + 10)
        cv2.putText(image, obj, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_response(prompt_content, base64_images, stream=False):
    client = OpenAI(api_key=MODEL_KEY)
    # client = OpenAI(base_url=MODEL_URL, api_key=MODEL_KEY)
    content = [
        {
            'type': 'text',
            'text': prompt_content,
        }
    ]
    for base64_image in base64_images:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        })
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'user',
                'content': content
            }
        ],
        # reasoning_effort='high',  # o3, gemini
        # extra_body={
        #     'thinking': {'type': 'enabled', 'budget_tokens': 12800}  # claude
        # },
        # extra_body={
        #     'enable_thinking': True  # qwen
        # },
        stream=stream,  # qwen
        temperature=0  # gpt-4o, qwen, gemini
    )
    if stream:
        reasoning_content = ''
        answer_content = ''
        is_answering = False
        print('\n' + '=' * 20 + 'Reasoning' + '=' * 20 + '\n')
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            if hasattr(delta, 'content') and delta.content:
                if not is_answering:
                    print('\n' + '=' * 20 + 'Response' + '=' * 20 + '\n')
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content
        return reasoning_content, answer_content
    else:
        return response.choices[0].message.content


def run():
    for root, dirs, files in os.walk(COMIC_DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                base64_image = image_to_base64(image_path)
                prompt_content = read_prompt(PROMPT_PATH)
                response = get_response(prompt_content, [base64_image])
                # reasoning, response = get_response(prompt_content, [base64_image], True)
                # print("Reasoning:", reasoning)
                print("Response:", response)
                response_json = extract_json(response)
                print('JSON response:', response_json)
                draw_bounding_boxes(image_path, response_json)


if __name__ == '__main__':
    run()
