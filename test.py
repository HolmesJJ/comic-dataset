import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def random_color():
    return random.random(), random.random(), random.random()


def run(image_path=None):
    boxes = {
        "Gohan": [38 / 376, 149 / 326, 92 / 376, 104 / 326],
        "Vegeta": [132 / 376, 2 / 326, 119 / 376, 306 / 326],
        "Kuririn": [202 / 376, 112 / 326, 167 / 376, 208 / 326],
        "desert": [3 / 376, 262 / 326, 370 / 376, 62 / 326],
        "cloud": [3 / 376, 207 / 326, 261 / 376, 45 / 326]
    }

    adjusted_boxes = {
        label: [x, 1 - y - h, w, h] for label, (x, y, w, h) in boxes.items()
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    if image_path and os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Bounding Boxes')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for label, (x, y, w, h) in adjusted_boxes.items():
        color = random_color()
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label=label)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, color='red', fontsize=10, ha='center', va='center')

    plt.legend([label for label in adjusted_boxes.keys()], loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run('data/21/14/10.jpg')
