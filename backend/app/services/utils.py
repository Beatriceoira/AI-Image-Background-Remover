import cv2
import numpy as np

# Define the utility functions
def load_image(path):
    # Load the image
    image = cv2.imread(path)
    return image