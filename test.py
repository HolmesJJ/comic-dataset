import os
import random
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from collections import defaultdict


load_dotenv()

INPUT_DIR = os.getenv('INPUT_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')


def random_color():
    return random.random(), random.random(), random.random()


def count_images(folder_path, unique_threshold=10, total_threshold=10):
    unique_images = set()
    image_labels = defaultdict(set)
    image_label_counts = defaultdict(int)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, usecols=['image_name', 'label_name'])
            for _, row in df.dropna(subset=['image_name', 'label_name']).iterrows():
                image_name = f"{file_name.split('.')[0]}_{row['image_name']}"
                unique_images.add(image_name)
                image_labels[image_name].add(row['label_name'])
                image_label_counts[image_name] += 1
    total_unique_images = len(unique_images)
    result_df = pd.DataFrame({
        'image_name': list(image_labels.keys()),
        'unique_label_count': [len(labels) for labels in image_labels.values()],
        'total_label_count': [image_label_counts[image] for image in image_labels.keys()]
    })
    filtered_df = result_df[
        (result_df['unique_label_count'] > unique_threshold) |
        (result_df['total_label_count'] > total_threshold)]
    print('Total images:', total_unique_images)
    print('Summary')
    print(filtered_df)
    print(len(filtered_df))


def run(csv_path, comic_dir):
    df = pd.read_csv(csv_path)
    page_name = os.path.basename(csv_path).split('.')[0]
    grouped = df.groupby('image_name')
    for image_name, group in grouped:
        image_name = str(image_name)
        adjusted_boxes = [
            {
                'label': row['label_name'],
                'x': row['bbox_x'] / row['image_width'],
                'y': (row['image_height'] - (row['bbox_y'] + row['bbox_height'])) / row['image_height'],
                'w': row['bbox_width'] / row['image_width'],
                'h': row['bbox_height'] / row['image_height']
            }
            for _, row in group.iterrows()
        ]
        image_path = os.path.join(comic_dir, page_name, image_name)
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            img_height, img_width = img.shape[:2]
            aspect_ratio = img_width / img_height
            fig_height = 7
            fig_width = fig_height * aspect_ratio
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')
        else:
            print(f'Image not found at path: {image_path}')
            return
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.set_title(image_name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for box in adjusted_boxes:
            color = random_color()
            rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(box['x'] + box['w'] / 2, box['y'] + box['h'] / 2, box['label'], color='red', fontsize=10, ha='center',
                    va='center')
        plt.legend([box['label'] for box in adjusted_boxes], loc='upper right')
        plt.show()


if __name__ == '__main__':
    count_images(os.path.join(OUTPUT_DIR, '01'))
    font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
    font_prop = FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    run(os.path.join(OUTPUT_DIR, '01', 'page_1.csv'), os.path.join(INPUT_DIR, '01'))
