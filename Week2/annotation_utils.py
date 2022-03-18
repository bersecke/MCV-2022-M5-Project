import sys

sys.path.insert(0, '/home/group05/MCV-2022-M5-Project')
from mots_tools.mots_common.io import load_txt

def annotation_txt_to_objs(path):
    return load_txt(path)