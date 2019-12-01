#!/usr/bin/python
from PIL import Image
import os, sys

path = "mami"
dirs = os.listdir(path)
#os.mkdir("mami_resized")

for dir in dirs:
    new_dir = os.path.join("mami_resized", dir)
    os.mkdir(new_dir)
    items = os.path.join(path, dir)

    for item in os.listdir(items):
        fullPath = os.path.join(items, item)

        if os.path.isfile(fullPath):
            try:
                im = Image.open(fullPath)
                imResize = im.resize((96,96), Image.ANTIALIAS)
                imResize = imResize.convert("RGB")
                imResize.save(os.path.join(new_dir, item), 'JPEG', quality=90)
            except:
                break
