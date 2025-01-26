import os

from dotenv import load_dotenv


load_dotenv()

INPUT_DIR = os.getenv('INPUT_DIR')


def add_prefix_to_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            parent_folder_name = os.path.basename(root)
            new_name = f'{parent_folder_name}_{file}'
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, new_name)
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')


if __name__ == '__main__':
    add_prefix_to_files(os.path.join(INPUT_DIR, '01'))
