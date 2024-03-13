import xml.etree.ElementTree as ET
import os
from PIL import Image
import random
import string
import shutil

file_xml = os.path.join("annotations.xml")
tree = ET.parse(file_xml)
root = tree.getroot()
i=0
for image_elem in root.findall("image"):
    image_filename = image_elem.get("name")
    image_path = os.path.join('image', image_elem.attrib['name'])
    img = Image.open(image_path)
    img = img.convert('RGB')
    for box in image_elem.iter("box"):
        label = box.get("label")
        xmin = int(float(box.get("xtl")))
        ymin = int(float(box.get("ytl")))
        xmax = int(float(box.get("xbr")))
        ymax = int(float(box.get("ybr")))
        object_img = img.crop((xmin, ymin, xmax, ymax))
        name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
<<<<<<< HEAD
        '''if i<50:
            object_path = os.path.join("datasets/aircraft", f'{name}.jpg')
            object_img.save(object_path)
        else:'''
        object_path = os.path.join("datasets/ship", f'{name}.jpg')
        object_img.save(object_path)
    i+=1

#air_len=len(os.listdir("datasets/aircraft"))
ship_len=len(os.listdir("datasets/ship"))
#air_border=round(air_len*0.8)
ship_border=round(ship_len*0.8)
#air_file=os.listdir("datasets/aircraft")
ship_file=os.listdir("datasets/ship")
j=0
'''for i in air_file:
=======
        if i<50:
            object_path = os.path.join("datasets/aircraft", f'{name}.jpg')
            object_img.save(object_path)
        else:
           object_path = os.path.join("datasets/ship", f'{name}.jpg')
           object_img.save(object_path)
    i+=1

air_len=len(os.listdir("datasets/aircraft"))
ship_len=len(os.listdir("datasets/ship"))
air_border=round(air_len*0.8)
ship_border=round(ship_len*0.8)
air_file=os.listdir("datasets/aircraft")
ship_file=os.listdir("datasets/ship")
j=0
for i in air_file:
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
    file_path = os.path.join("datasets/aircraft", i)
    if j<=air_border:
        shutil.move(file_path, "datasets/train/aircraft")
    else:
        shutil.move(file_path, "datasets/test/aircraft")
<<<<<<< HEAD
    j+=1'''
=======
    j+=1
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
j=0
for i in ship_file:
    file_path = os.path.join("datasets/ship", i)
    if j<=ship_border:
        shutil.move(file_path, "datasets/train/ship")
    else:
        shutil.move(file_path, "datasets/test/ship")
    j+=1
<<<<<<< HEAD
#shutil.rmtree("datasets/aircraft")
=======
shutil.rmtree("datasets/aircraft")
>>>>>>> c83e9f2f6453f4b5b243739948ea80753d2bd72e
shutil.rmtree("datasets/ship")
