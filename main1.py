import os
import re
import cv2
import json
import base64

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

COMIC_DIR = os.getenv('COMIC_DIR')
MODEL = os.getenv('MODEL')
OPENAI_KEY = os.getenv('OPENAI_KEY')
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
    for root, dirs, files in os.walk(COMIC_DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                base64_image = image_to_base64(image_path)
                prompt_content = read_prompt(PROMPT_PATH)
                response = get_response(prompt_content, base64_image)
                response_json = extract_json(response)
                print('JSON response:', response_json)
                draw_bounding_boxes(image_path, response_json)


if __name__ == '__main__':
    run()
