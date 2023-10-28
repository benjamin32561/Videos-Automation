from os.path import join, exists
from os import listdir, makedirs
from random import choice

def CreateFolder(folder_path):
    if not exists(folder_path):
        makedirs(folder_path)

def SelectImageByKeyphrase(keyphrase:str, images_path:str):
    folder_path = join(images_path,keyphrase)
    files = listdir(folder_path)
    file_paths = [join(folder_path, file_name) for file_name in files]

    # Get a random file path
    return choice(file_paths)