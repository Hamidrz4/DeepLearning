import random
from PIL import Image, ImageDraw
import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg

def match_images_with_annotations(image_directory, annotations_file):
    # Read the annotations file
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Create a dictionary to store matched image and annotation data
    matched_data = {}

    # Iterate over images in the image directory
    for image_info in data['images']:
        image_id = image_info['id']
        image_file_name = image_info['file_name']
        image_path = os.path.join(image_directory, image_file_name)

        # Get annotations for the current image
        image_annotations = [annotation for annotation in data['annotations'] if
                             annotation['image_id'] == image_id]

        # Check if there are annotations available for the current image
        if image_annotations:
            # Get the category/class for the current image
            category_id = image_annotations[0]['category_id']  # Assuming each image has only one annotation
            category_info = [category for category in data['categories'] if
                             category['id'] == category_id]
            category_name = category_info[0]['name']

            # Add the image path, annotations, and category to the matched data dictionary
            matched_data[image_path] = {'annotations': image_annotations, 'category': category_name}
        else:
            print(f"No annotations found for image: {image_file_name}")

    return matched_data


def display_images_with_annotations(image_annotations_dict, num_images):
    # Convert the dictionary to a list of tuples (image path, annotations)
    image_annotations_list = list(image_annotations_dict.items())

    # Shuffle the list
    random.shuffle(image_annotations_list)

    # Display a specific number of images with their annotations and categories
    for i, (image_path, annotations_data) in enumerate(image_annotations_list[:num_images]):
        print(f"Displaying Image {i + 1}/{num_images}")
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for annotation in annotations_data['annotations']:
            segmentation = annotation['segmentation']
            # Flatten the segmentation points list
            flattened_segmentation = [point for sublist in segmentation for point in sublist]
            draw.polygon(flattened_segmentation, outline="red")

        # Get the category name for the image
        category_name = annotations_data['category']
        # Add category information to the image display
        draw.text((10, 10), f"Class -> {category_name}", fill="white")

        image.show()




def create_df(dictionary):
    # Convert the matched_data dictionary into a DataFrame
    data = []
    for image_path, info in dictionary.items():
        data.append((image_path, info['category']))
    df = pd.DataFrame(data, columns=['image_path', 'category'])
    return df

def get_df(dataframe):
    idg_object = idg(rescale=1./255)
    data_generator = idg_object.flow_from_dataframe(
        dataframe,
        target_size=(224,224),
        batch_size=32,
        x_col='image_path',
        y_col='category',
        class_mode="binary",
        color_mode="grayscale"
    )
    return data_generator