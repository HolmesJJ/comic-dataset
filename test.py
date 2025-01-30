import os
import random
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


load_dotenv()

INPUT_DIR = os.getenv('INPUT_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')


def random_color():
    return random.random(), random.random(), random.random()


def count_images(folder_path):
    unique_images = set()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, usecols=['image_name'])
            image_names = df['image_name'].dropna().unique()
            image_names = [file_name.split('.')[0] + '_' + image_name for image_name in image_names]
            unique_images.update(image_names)
    return len(unique_images)


def run(csv_path, comic_dir):
    df = pd.read_csv(csv_path)
    page_name = os.path.basename(csv_path).split('.')[0]
    grouped = df.groupby('image_name')
    for image_name, group in grouped:
        image_name = str(image_name)
        adjusted_boxes = [
            {
                "label": row['label_name'],
                "x": row['bbox_x'] / row['image_width'],
                "y": (row['image_height'] - (row['bbox_y'] + row['bbox_height'])) / row['image_height'],
                "w": row['bbox_width'] / row['image_width'],
                "h": row['bbox_height'] / row['image_height']
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
    print(count_images(os.path.join(OUTPUT_DIR, '01')))
    font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
    font_prop = FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    run(os.path.join(OUTPUT_DIR, '01', 'page_96.csv'), os.path.join(INPUT_DIR, '01'))
