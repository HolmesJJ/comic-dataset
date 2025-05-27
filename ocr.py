import os
import re
import csv
import cv2
import base64
import numpy as np
import pandas as pd

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter


load_dotenv()

COMIC = os.getenv('COMIC')
COMIC_DIR = os.getenv('COMIC_DIR')
DIALOGUE_DIR = os.path.join(os.getenv('DIALOGUE_DIR'), COMIC)
MODEL = os.getenv('GPT_MODEL')  # GPT_MODEL, QWEN_MODEL, CLAUDE_MODEL, GEMINI_MODEL
MODEL_KEY = os.getenv('GPT_KEY')  # GPT_KEY, QWEN_KEY, CLAUDE_KEY, GEMINI_KEY
MODEL_URL = os.getenv('QWEN_URL')  # QWEN_URL, CLAUDE_URL, GEMINI_URL
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


def sort_page(root):
    match = re.search(r'page_(\d+)', root)
    return int(match.group(1)) if match else float('inf')


def sort_image(file):
    match = re.search(r'(\d+)', file)
    return int(match.group(1)) if match else float('inf')


def sort_csv(comic):
    csv_path = os.path.join(DIALOGUE_DIR, f'{comic}.csv')
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=['Page', 'Image'],
                               key=lambda x: x.map(sort_page) if x.name == 'Page' else x.map(sort_image))
    df_sorted.to_csv(csv_path, index=False)


def clean_csv(comic):
    csv_path = os.path.join(DIALOGUE_DIR, f'{comic}.csv')
    df = pd.read_csv(csv_path)
    df['Dialogue'] = df['Dialogue'].replace('<Dialogue>', '', regex=False)
    df['Dialogue'] = df['Dialogue'].str.replace(r'[<>]', '', regex=True)
    df['Dialogue'] = df['Dialogue'].str.replace('…', '...', regex=False)
    df['Dialogue'] = df['Dialogue'].str.replace(',', '，', regex=False)
    df['Dialogue'] = df['Dialogue'].str.replace('?', '？', regex=False)
    df['Dialogue'] = df['Dialogue'].str.replace('!', '！', regex=False)
    df['Dialogue'] = df['Dialogue'].str.strip()
    df[['Dialogue', 'Character']] = df[['Dialogue', 'Character']].replace('', np.nan)
    df.dropna(subset=['Dialogue', 'Character'], how='all', inplace=True)
    df.to_csv(csv_path, index=False)


def run(comic):
    csv_path = os.path.join(DIALOGUE_DIR, f'{comic}.csv')
    file_exists = os.path.exists(csv_path)
    processed_files = set()
    if file_exists:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:
                    processed_files.add((row[0], row[1]))
    with open(os.path.join(DIALOGUE_DIR, f'{comic}.csv'), mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(['Page', 'Image', 'Character', 'Dialogue'])
        root_dirs = []
        for root, dirs, files in os.walk(os.path.join(COMIC_DIR, comic)):
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
                    response = get_response(prompt_content, [base64_image])
                    # reasoning, response = get_response(prompt_content, [base64_image], True)
                    # print("Reasoning:", reasoning)
                    print("Response:", response)
                    dialogues = response.strip().split('\n')
                    for dialogue in dialogues:
                        content = dialogue.strip().split(',')
                        if len(content) == 2 and content[1]:
                            row = [folder_name, file, content[0], content[1]]
                            csv_writer.writerow(row)
                            print(row)


def compare(comic):
    csv1_path = os.path.join(DIALOGUE_DIR, f'{comic}.csv')
    csv2_path = os.path.join(DIALOGUE_DIR, f'{comic}_updated.csv')
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df_diff = pd.concat([df1, df2]).drop_duplicates(keep=False)
    summary = {
        'total_rows_csv1': len(df1),
        'total_rows_csv2': len(df2),
        'different_rows': len(df_diff)
    }
    print('Differences between CSV files:')
    print(df_diff)
    print('\nSummary:')
    print(summary)


def check_label_unique():
    label_counter = Counter()
    all_csv_files = []
    for root, dirs, files in os.walk(DIALOGUE_DIR, topdown=True):
        for file in files:
            if '_updated' in file and file.endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    for file_path in tqdm(all_csv_files, desc='Processing CSVs'):
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] != 4:
                print(f'Invalid column count in {file_path}: {df.shape[1]} columns')
            labels = df['Character'].dropna()
            label_counter.update(labels)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
    output_df = pd.DataFrame(
        sorted(label_counter.items(), key=lambda x: x[0]),
        columns=['character', 'count']
    )
    output_path = os.path.join(DIALOGUE_DIR, 'character_summary.csv')
    output_df.to_csv(output_path, index=False)
    print(f'Characters with counts saved to: {output_path}')


if __name__ == '__main__':
    # compare('01')
    # run('01')
    # for i in range(1, 43):
    #     clean_csv(f'{i:02d}')
    #     sort_csv(f'{i:02d}')
    check_label_unique()
