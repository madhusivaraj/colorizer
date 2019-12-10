import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import PIL
import cv2
import os


#make sure to have pictures in the same directory as this file

def color_to_greyscale(rgb):
    return np.dot(rgb[...,:3], [0.21, 0.72, 0.07])


def save_grayscale(rgb):
    img = mpimg.imread(rgb)
    img_grayscale = color_to_greyscale(img)
    index= rgb.index(".jpg")
    cv2.imwrite(rgb[:index]+"_bw"+rgb[index:], img_grayscale)

"""for entry in os.scandir():
    if entry.name.endswith(".jpg") and "_bw" not in entry.name and entry.is_file():
        save_grayscale(entry.name)"""

#Scales rgb value for black/white image
img_bw=cv2.imread('image_0001_bw.jpg',1)
new_img_bw_scaled=img_bw/255.0
plt.imshow(new_img_bw_scaled)
plt.show()

#scales rgb values for colored image. 
img =cv2.imread('image_0001.jpg',1)
new_img_scaled=img/255.0
plt.imshow(new_img_scaled)
plt.show()


