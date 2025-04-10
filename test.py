import os
import random
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from matplotlib import rcParams
from matplotlib.patches import Patch
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
    calculate_price(total_unique_images, result_df)


def calculate_price(total_unique_images, df):
    count_10_15 = df[(df['total_label_count'] > 10) & (df['total_label_count'] <= 15)].shape[0]
    count_15_plus = df[df['total_label_count'] > 15].shape[0]
    count_default = total_unique_images - count_10_15 - count_15_plus
    total_cost = count_default * 0.2 + count_10_15 * 0.3 + count_15_plus * 0.4
    formula = f'({count_default} * 0.2) + ({count_10_15} * 0.3) + ({count_15_plus} * 0.4)'
    print(f'Total Images: {total_unique_images}')
    print(f'Images with total_label_count > 15: {count_15_plus}')
    print(f'Images with 10 < total_label_count ≤ 15: {count_10_15}')
    print(f'Images with total_label_count ≤ 10: {count_default}')
    print(f'Total Cost: {formula} = {total_cost:.2f}')

    return total_cost


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
        labels_colors = {box['label']: random_color() for box in adjusted_boxes}
        for box in adjusted_boxes:
            color = labels_colors[box['label']]
            rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(box['x'] + box['w'] / 2, box['y'] + box['h'] / 2, box['label'], color='red', fontsize=10, ha='center',
                    va='center')
        handles = [Patch(edgecolor=color, facecolor='none', linewidth=2, label=label)
                   for label, color in labels_colors.items()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
        plt.tight_layout()
        plt.show()


def file_exists(row, folder_path):
    image_path = os.path.join(folder_path, row['page'], row['image_name'])
    return os.path.exists(image_path)


def check(comic):
    input_images = set()
    for root, dirs, files in os.walk(os.path.join(INPUT_DIR, comic), topdown=True):
        for file in files:
            file_path = os.path.join(root, file)
            image_name = os.path.basename(file_path)
            page = os.path.basename(root)
            input_images.add((image_name, page))
    csv_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, comic))]
    df = pd.DataFrame()
    for file in csv_files:
        file_path = os.path.join(os.path.join(OUTPUT_DIR, comic), file)
        page_df = pd.read_csv(file_path, usecols=['image_name', 'image_width', 'image_height'])
        page_df = page_df.drop_duplicates()
        page = os.path.basename(file_path).split('.')[0]
        page_df['page'] = page
        df = pd.concat([df, page_df], ignore_index=True)
    df_images = set(zip(df['image_name'], df['page']))
    missing_from_df = input_images - df_images
    missing_from_input = df_images - input_images
    for image_name, page in sorted(missing_from_df, key=lambda x: (int(x[1].split('_')[-1]), x[0])):
        print(f'{page}/{image_name}')
    print('-' * 10)
    for image_name, page in sorted(missing_from_input, key=lambda x: (int(x[1].split('_')[-1]), x[0])):
        print(f'{page}/{image_name}')


if __name__ == '__main__':
    # for i in range(1, 43):
    #     check(f'{i:02d}')
    count_images(os.path.join(OUTPUT_DIR, '01'))
    font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
    font_prop = FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    run(os.path.join(OUTPUT_DIR, '01', 'page_1.csv'), os.path.join(INPUT_DIR, '01'))
