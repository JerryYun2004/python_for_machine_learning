
import kagglehub

# Download latest version
path = kagglehub.dataset_download("gunhcolab/object-detection-dataset-standard-52card-deck")

print("Path to dataset files:", path)

import os
import cv2 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

#paths
images_path = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train/train"
annotations_path = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train/train"

#Load one image and its annotation
image_file = "/1.jpg"
annotation_file = "/1.xml"
image_path = images_path + image_file
annotation_path = images_path + annotation_file

#Read image
img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Parse annotation
tree = ET.parse(annotation_path)
root = tree.getroot()

for obj in root.findall('object'):
    name = obj.find('name').text
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)

    #Draw rectange and label
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(img, name, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#Show Result
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')
plt.title("Card Detection Preview")
plt.show()
