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


def random_color():
    return random.random(), random.random(), random.random()


def run(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    adjusted_boxes = [
        {
            "label": row['label_name'],
            "x": row['bbox_x'] / row['image_width'],
            "y": (row['image_height'] - (row['bbox_y'] + row['bbox_height'])) / row['image_height'],
            "w": row['bbox_width'] / row['image_width'],
            "h": row['bbox_height'] / row['image_height']
        }
        for _, row in df.iterrows()
    ]
    image_name = df['image_name'].iloc[0]
    image_path = os.path.join(image_dir, image_name.rsplit('_', 1)[0], image_name)
    fig, ax = plt.subplots(figsize=(12, 12))
    if image_path and os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')
    else:
        print(f'Image not found at path: {image_path}')
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Bounding Boxes')
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
    font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
    font_prop = FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    run(os.path.join(INPUT_DIR, 'labels.csv'), os.path.join(INPUT_DIR, '01'))
