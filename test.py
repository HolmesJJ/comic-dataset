import os
import random
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from collections import defaultdict


load_dotenv()

COMIC = os.getenv('COMIC')
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
OBJECT_DIR = os.path.join(os.getenv('OBJECT_DIR'), COMIC)
COMIC_ANIME_DIR = os.path.join(os.getenv('COMIC_ANIME_DIR'), COMIC)

FONT_PATH = 'C:\\Windows\\Fonts\\SimHei.ttf'
FONT_PROP = FontProperties(fname=FONT_PATH)
rcParams['font.family'] = FONT_PROP.get_name()


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


def create_voc_xml(image_name, image_width, image_height, objects):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = ''
    file_name = ET.SubElement(annotation, 'filename')
    file_name.text = image_name
    path = ET.SubElement(annotation, 'path')
    path.text = ''
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unspecified'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_width)
    height = ET.SubElement(size, 'height')
    height.text = str(image_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for obj in objects:
        obj_element = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj_element, 'name')
        name.text = obj['label_name']
        pose = ET.SubElement(obj_element, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj_element, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj_element, 'difficult')
        difficult.text = '0'
        bnd_box = ET.SubElement(obj_element, 'bndbox')
        x_min = ET.SubElement(bnd_box, 'xmin')
        x_min.text = str(int(obj['bbox_x']))
        y_min = ET.SubElement(bnd_box, 'ymin')
        y_min.text = str(int(obj['bbox_y']))
        x_max = ET.SubElement(bnd_box, 'xmax')
        x_max.text = str(int(obj['bbox_x'] + obj['bbox_width']))
        y_max = ET.SubElement(bnd_box, 'ymax')
        y_max.text = str(int(obj['bbox_y'] + obj['bbox_height']))
    return ET.ElementTree(annotation)


def save_pretty_xml(tree, output_path):
    xml_str = ET.tostring(tree.getroot(), encoding='utf-8')
    parsed = minidom.parseString(xml_str)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed.toprettyxml(indent='\t'))


def csv_to_voc(comic):
    comic_dir = os.path.join(OBJECT_DIR, comic)
    for file_name in os.listdir(comic_dir):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(comic_dir, file_name)
            object_dir = os.path.join(comic_dir, os.path.splitext(file_name)[0])
            df = pd.read_csv(csv_file)
            grouped = df.groupby('image_name')
            if not os.path.exists(object_dir):
                os.makedirs(object_dir)
            for image_name, group in grouped:
                image_width = int(group['image_width'].iloc[0])
                image_height = int(group['image_height'].iloc[0])
                xml_tree = create_voc_xml(image_name, image_width, image_height, group.to_dict(orient='records'))
                save_pretty_xml(xml_tree, os.path.join(object_dir, str(image_name).replace('.jpg', '.xml')))


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
            fig_width = fig_height * aspect_ratio + 2  # 多留出 legend 空间
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(1, 2, width_ratios=[aspect_ratio, 0.4])  # 第二列为 legend
            ax = fig.add_subplot(gs[0])
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
            rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=2,
                                 edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(box['x'] + box['w'] / 2, box['y'] + box['h'] / 2, box['label'],
                    color='red', fontsize=10, ha='center', va='center')
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis('off')
        handles = [Patch(edgecolor=color, facecolor='none', linewidth=2, label=label)
                   for label, color in labels_colors.items()]
        legend_ax.legend(handles=handles, loc='center left')
        plt.show()


def file_exists(row, folder_path):
    image_path = os.path.join(folder_path, row['page'], row['image_name'])
    return os.path.exists(image_path)


def check_missing(comic, checked):
    input_images = set()
    input_image_sizes = {}
    for root, dirs, files in os.walk(os.path.join(COMIC_DIR, comic), topdown=True):
        for file in files:
            file_path = os.path.join(root, file)
            image_name = os.path.basename(file_path)
            page = os.path.basename(root)
            input_images.add((image_name, page))
            try:
                with Image.open(file_path) as img:
                    input_image_sizes[(image_name, page)] = img.size
            except Exception as e:
                print(f'Error image {file_path}: {e}')
    csv_files = [f for f in os.listdir(os.path.join(OBJECT_DIR, f'{comic}(已检查)' if checked else comic))]
    df = pd.DataFrame()
    for file in csv_files:
        file_path = os.path.join(os.path.join(OBJECT_DIR, f'{comic}(已检查)' if checked else comic), file)
        page_df = pd.read_csv(file_path, usecols=['image_name', 'image_width', 'image_height'])
        page_df = page_df.drop_duplicates()
        page = os.path.basename(file_path).split('.')[0]
        page_df['page'] = page
        df = pd.concat([df, page_df], ignore_index=True)
    df_images = set(zip(df['image_name'], df['page']))
    missing_from_df = input_images - df_images
    missing_from_input = df_images - input_images
    # print(f'{comic} 没有标记对象的图：')
    # for image_name, page in sorted(missing_from_df, key=lambda x: (int(x[1].split('_')[-1]), x[0])):
    #     print(f'{page}/{image_name}')
    print(f'{comic} 标记了但找不到对应的图：')
    for image_name, page in sorted(missing_from_input, key=lambda x: (int(x[1].split('_')[-1]), x[0])):
        print(f'{page}/{image_name}')
    print(f'{comic} 尺寸不一致的图：')
    for _, row in df.iterrows():
        key = (row['image_name'], row['page'])
        if key in input_image_sizes:
            actual_w, actual_h = input_image_sizes[key]
            csv_w, csv_h = int(row['image_width']), int(row['image_height'])
            if actual_w != csv_w or actual_h != csv_h:
                print(f"{row['page']}/{row['image_name']}：CSV=({csv_w},{csv_h})，实际=({actual_w},{actual_h})")


def check_label_unique():
    label_counter = Counter()
    all_csv_files = []
    for root, dirs, files in os.walk(OBJECT_DIR, topdown=True):
        for file in files:
            if file.endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))
    for file_path in tqdm(all_csv_files, desc='Processing CSVs'):
        try:
            df = pd.read_csv(file_path, usecols=['label_name'])
            labels = df['label_name'].dropna()
            label_counter.update(labels)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
    output_df = pd.DataFrame(
        sorted(label_counter.items(), key=lambda x: x[0]),
        columns=['label_name', 'count']
    )
    output_path = os.path.join(OBJECT_DIR, 'label_summary.csv')
    output_df.to_csv(output_path, index=False)
    print(f'Label names with counts saved to: {output_path}')


if __name__ == '__main__':
    # for i in range(1, 17):
    #     check_missing(f'{i:02d}', True)
    # for i in range(17, 43):
    #     check_missing(f'{i:02d}', False)
    check_label_unique()
    # csv_to_voc('01(已检查)')
    # count_images(os.path.join(OBJECT_DIR, '01(已检查)'))
    # run(os.path.join(OBJECT_DIR, '01(已检查)', 'page_6.csv'), os.path.join(COMIC_DIR, '01'))
