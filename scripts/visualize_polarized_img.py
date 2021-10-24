import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        images = np.load(f)
        image1 = cv2.cvtColor(images[:, :, 0:3], cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(images[:, :, 3:6], cv2.COLOR_RGB2BGR)
        image3 = cv2.cvtColor(images[:, :, 6:9], cv2.COLOR_RGB2BGR)
        cv2.imwrite(name + '_1.jpg', image1 * 255)
        cv2.imwrite(name + '_2.jpg', image2 * 255)
        cv2.imwrite(name + '_3.jpg', image3 * 255)
