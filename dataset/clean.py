import os
import shutil

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions

def delete_subfolders_with_one_image(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for d in dirs:
            subfolder_path = os.path.join(root, d)
            image_files = [f for f in os.listdir(subfolder_path) if is_image_file(f)]
            if len(image_files) == 1:
                print(f"Deleting subfolder: {subfolder_path} with 1 image file")
                shutil.rmtree(subfolder_path)

# Example usage:
delete_subfolders_with_one_image('C:/Users\Mohsen\GitProjects\eval_face_models\dataset\lfw-deepfunneled - Copy')
