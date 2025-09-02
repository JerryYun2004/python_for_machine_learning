import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def load_images_and_annotations(folder_path, max_images=None):
    dataset = []
    supported_image_exts = (".jpg", ".png")

    # Get all files with image extensions that also have a matching .xml file
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_image_exts)
        and os.path.exists(os.path.join(folder_path, os.path.splitext(f)[0] + ".xml"))
    ]

    if max_images:
        image_files = image_files[:max_images]

    for filename in sorted(image_files):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(folder_path, filename)
        annotation_path = os.path.join(folder_path, base_name + ".xml")

        img = cv2.imread(image_path)
        annotations = []

        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # Draw bounding box and label
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, name, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                annotations.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})

            dataset.append((img, annotations))

        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    return dataset


if __name__ == "__main__":
    folder_path = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train/train"
    dataset = load_images_and_annotations(folder_path)

    for idx, (img, ann) in enumerate(dataset, start=1):
        print(f"Image {idx}: {ann}")