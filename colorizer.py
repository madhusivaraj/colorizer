import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def color_to_greyscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

img = mpimg.imread('img/yosemite.jpeg')
plt.imshow(img)
#plt.show()
img_grayscale = color_to_greyscale(img)
plt.imshow(img_grayscale, cmap = plt.get_cmap('gray'))
# plt.savefig(new_path)
plt.show()
