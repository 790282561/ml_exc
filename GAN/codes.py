import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
dirlist = os.listdir('pics\\generated_num')
for i, j in enumerate(dirlist):
    img_full = np.ones((28, 28))
    img = mpimg.imread("pics\\generated_num\\" + j)
    img = img_full - img

    plt.subplot(1, 13, i + 1)
    plt.imshow(img, cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.show()
