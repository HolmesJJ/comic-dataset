import os
import re
import cv2
import json
import base64
import random
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

COMIC = os.getenv('COMIC')
COMIC_ANIME_DIR = os.path.join(os.getenv('COMIC_ANIME_DIR'), COMIC, '1')
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
DIALOGUE_DIR = os.path.join(os.getenv('DIALOGUE_DIR'), COMIC)
OBJECT_DIR = os.path.join(os.getenv('OBJECT_DIR'), COMIC)
MODEL = os.getenv('MODEL')
OPENAI_KEY = os.getenv('OPENAI_KEY')
PROMPT_PATH = os.getenv('PROMPT2_PATH')


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


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


def show_bboxes(image_path, object_df, max_width=768, max_height=768):
    label_colors = {}
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img_height, img_width = img.shape[:2]
    for _, row in object_df.iterrows():
        x = int(row['x'] * img_width)
        y = int(row['y'] * img_height)
        w = int(row['w'] * img_width)
        h = int(row['h'] * img_height)
        label = row['object']
        if label not in label_colors:
            label_colors[label] = random_color()
        color = label_colors[label]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=20)
    scale = min(max_width / img_width, max_height / img_height, 1.0)
    if scale < 1.0:
        img_display = cv2.resize(img, (int(img_width * scale), int(img_height * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_display = img
    cv2.imshow("Detected Objects", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_objects(comic_block_ids):
    result = ''
    total_blocks = len(comic_block_ids)
    for idx, comic_block_id in enumerate(comic_block_ids):
        parts = comic_block_id.split('_')
        comic_id = parts[0]
        page_folder = f'page_{parts[2]}'
        image_path = None
        image_path_jpg = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.jpg')
        image_path_png = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.png')
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_png):
            image_path = image_path_png
        if not image_path:
            print(image_path)
            raise ValueError('Image path is missing or invalid.')
        object_path = os.path.join(OBJECT_DIR, f'{comic_id}(已检查)', f'{page_folder}.csv')
        if not os.path.exists(object_path):
            print(object_path)
            raise ValueError('Object path is missing or invalid.')
        object_df = pd.read_csv(object_path)
        object_df = object_df[(object_df['image_name'] == os.path.basename(image_path))]
        object_df['object'] = object_df['label_name']
        object_df['x'] = object_df['bbox_x'] / object_df['image_width']
        object_df['y'] = object_df['bbox_y'] / object_df['image_height']
        object_df['w'] = object_df['bbox_width'] / object_df['image_width']
        object_df['h'] = object_df['bbox_height'] / object_df['image_height']
        object_df = object_df[['object', 'x', 'y', 'w', 'h']]
        # show_bboxes(image_path, object_df)
        panel_title = f'CSV #1 Panel {idx + 1}' if idx < total_blocks - 1 else 'CSV #1 Last Panel'
        result += f'{panel_title}\n{object_df.to_csv(index=False)}\n'
    return result


def get_dialogues(comic_block_ids):
    result = ''
    total_blocks = len(comic_block_ids)
    for idx, comic_block_id in enumerate(comic_block_ids):
        parts = comic_block_id.split('_')
        comic_id = parts[0]
        page_folder = f'page_{parts[2]}'
        image_path = None
        image_path_jpg = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.jpg')
        image_path_png = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.png')
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_png):
            image_path = image_path_png
        if not image_path:
            print(image_path)
            raise ValueError('Image path is missing or invalid.')
        dialogue_path = os.path.join(DIALOGUE_DIR, f'{comic_id}_updated(已检查).csv')
        if not os.path.exists(dialogue_path):
            print(dialogue_path)
            raise ValueError('Dialogue path is missing or invalid.')
        dialogue_df = pd.read_csv(dialogue_path)
        dialogue_df = dialogue_df[
            (dialogue_df['Page'] == page_folder) &
            (dialogue_df['Image'] == os.path.basename(image_path))
        ].fillna('旁白')
        dialogue_df = dialogue_df.rename(columns={'Character': 'object', 'Dialogue': 'dialogue'})
        dialogue_df = dialogue_df[['object', 'dialogue']]
        panel_title = f'CSV #1 Panel {idx + 1}' if idx < total_blocks - 1 else 'CSV #1 Last Panel'
        result += f'{panel_title}\n{dialogue_df.to_csv(index=False)}\n'
    return result


def run():
    comic_anime_files = sorted([f for f in os.listdir(COMIC_ANIME_DIR) if f.endswith('.csv')])
    all_comic_blocks = []
    for comic_anime_file in comic_anime_files:
        comic_anime_path = os.path.join(COMIC_ANIME_DIR, comic_anime_file)
        comic_anime_df = pd.read_csv(comic_anime_path)
        for index, row in comic_anime_df.iterrows():
            current_comic_block_id = row['Comic Block ID']
            if pd.isna(current_comic_block_id):
                continue
            all_comic_blocks.append(current_comic_block_id)
            start_idx = max(0, len(all_comic_blocks) - 6)
            pre_comic_block_ids = all_comic_blocks[start_idx:-1]
            comic_block_ids = pre_comic_block_ids
            comic_block_ids.append(current_comic_block_id)
            objects = get_objects(comic_block_ids)
            print(objects)
            dialogues = get_dialogues(comic_block_ids)
            print(dialogues)
            # prompt_content = read_prompt(PROMPT_PATH).format(COMIC, len(pre_current_ids))
            # print(prompt_content)

    # for root, dirs, files in os.walk(COMIC_DIR):
    #     for file in files:
    #         if file.endswith('.png') or file.endswith('.jpg'):
    #             image_path = os.path.join(root, file)
    #             base64_image = image_to_base64(image_path)
    #             layout = read_layout(layout_path, image_path[len(COMIC_DIR)+1:].replace('\\', '/'))
    #             layout_json = json.dumps(layout, indent=4)
    #             prompt_content = read_prompt(PROMPT_PATH).format(COMIC, 6)
    #             print(prompt_content)
    #             # TODO: 增加前情回顾
    #             prompt_content = prompt_content + '```json\n' + layout_json + '\n```'
    #             response = get_response(prompt_content, base64_image)
    #             response_json = extract_json(response)
    #             print('JSON response:', response_json)


if __name__ == '__main__':
    run()
