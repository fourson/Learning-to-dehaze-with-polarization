import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        img = np.load(f)
        if img.ndim == 3:
            if img.shape[2] == 3:
                # RGB image (H, W, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # ============================================
                # white balance
                B, G, R = cv2.split(img)
                B_mean = B.mean()
                G_mean = G.mean()
                R_mean = R.mean()
                KB = (B_mean + G_mean + R_mean) / (3 * B_mean)
                KG = (B_mean + G_mean + R_mean) / (3 * G_mean)
                KR = (B_mean + G_mean + R_mean) / (3 * R_mean)
                B = B * KB
                G = G * KG
                R = R * KR
                img = cv2.merge([B, G, R])
                # multiply 1.25 for better visualization
                img = img * 1.25
                # ============================================
            elif img.shape[2] == 1:
                # grayscale image (H, W, 1)
                img = img[:, :, 0]

        img = np.clip(img, a_min=0, a_max=1)
        cv2.imwrite(name + '.jpg', img * 255)
