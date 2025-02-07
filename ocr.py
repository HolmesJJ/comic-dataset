import os
import re
import csv
import cv2
import base64
import numpy as np
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

INPUT_DIR = os.getenv('INPUT_DIR')
OCR_DIR = os.getenv('OCR_DIR')
MODEL = os.getenv('MODEL')
OPENAI_KEY = os.getenv('OPENAI_KEY')
PROMPT_PATH = os.getenv('PROMPT4_PATH')


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def compress_image(image_path, max_size=512):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes(), resized_img


def image_to_base64(image_path):
    compressed_image, resized_img = compress_image(image_path)
    base64_str = base64.b64encode(compressed_image).decode('utf-8')
    decoded_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(decoded_data, np.uint8)
    decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return base64_str, decoded_img


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


def sort_page(root):
    match = re.search(r'page_(\d+)', root)
    return int(match.group(1)) if match else float('inf')


def sort_image(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')


def sort_csv(comic):
    csv_path = os.path.join(OCR_DIR, f'{comic}.csv')
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=['Page', 'Image'],
                               key=lambda x: x.map(sort_page) if x.name == 'Page' else x.map(sort_image))
    df_sorted.to_csv(csv_path, index=False)


def clean_csv(comic):
    csv_path = os.path.join(OCR_DIR, f'{comic}.csv')
    df = pd.read_csv(csv_path)
    df['Dialogue'] = df['Dialogue'].replace('<Dialogue>', '', regex=False)
    df['Dialogue'] = df['Dialogue'].str.replace(r'[<>]', '', regex=True)
    df['Dialogue'] = df['Dialogue'].str.replace('…', '...', regex=False)
    df.dropna(subset=['Dialogue', 'Character'], how='all', inplace=True)
    df.to_csv(csv_path, index=False)


def run(comic):
    csv_path = os.path.join(OCR_DIR, f'{comic}.csv')
    file_exists = os.path.exists(csv_path)
    processed_files = set()
    if file_exists:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:
                    processed_files.add((row[0], row[1]))
    with open(os.path.join(OCR_DIR, f'{comic}.csv'), mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(['Page', 'Image', 'Character', 'Dialogue'])
        root_dirs = []
        for root, dirs, files in os.walk(os.path.join(INPUT_DIR, comic)):
            root_dirs.append((root, dirs, files))
        root_dirs.sort(key=lambda x: sort_page(x[0]))
        for root, dirs, files in root_dirs:
            files.sort(key=sort_image)
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    folder_name = os.path.basename(root)
                    if (folder_name, file) in processed_files:
                        print(f'Skipping already processed file: {folder_name}/{file}')
                        continue
                    image_path = os.path.join(root, file)
                    base64_image, decoded_img = image_to_base64(image_path)
                    # cv2.imshow(file, decoded_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    prompt_content = read_prompt(PROMPT_PATH)
                    response = get_response(prompt_content, base64_image)
                    dialogues = response.strip().split('\n')
                    for dialogue in dialogues:
                        content = dialogue.strip().split(',')
                        if len(content) == 2 and content[1]:
                            row = [folder_name, file, content[0], content[1]]
                            csv_writer.writerow(row)
                            print(row)


if __name__ == '__main__':
    run('01')
    clean_csv('01')
    sort_csv('01')
