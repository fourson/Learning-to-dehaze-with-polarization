import os
import numpy as np
import cv2


def ensure_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def preprocess_raw(f):
    with open(f, 'rb') as raw_file:
        img = np.fromfile(raw_file, dtype=np.uint8)
        img = img.reshape((1024, 1224))
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        img = img[0::2, 0::2, :]
        img = np.float32(img)
        img /= 255.
        img = img[:, 2: -2, :]
        return img


input_dir = '../lucidImage'
output_dir = '../data/real'

I_alpha_dir = os.path.join(output_dir, 'I_alpha')
I_hat_dir = os.path.join(output_dir, 'I_hat')
delta_I_hat_dir = os.path.join(output_dir, 'delta_I_hat')

ensure_dir(output_dir)
ensure_dir(I_alpha_dir)
ensure_dir(I_hat_dir)
ensure_dir(delta_I_hat_dir)

alpha = np.array([0, np.pi / 4, np.pi / 2], dtype=np.float32)  # (3,)

for file_name in os.listdir(input_dir):
    if file_name.endswith('_0°.raw'):
        name = file_name.split('_0°.raw')[0]
        img_0_path = os.path.join(input_dir, name + '_0°.raw')
        img_0 = preprocess_raw(img_0_path)

        img_45_path = os.path.join(input_dir, name + '_45°.raw')
        img_45 = preprocess_raw(img_45_path)

        img_90_path = os.path.join(input_dir, name + '_90°.raw')
        img_90 = preprocess_raw(img_90_path)

        # not required
        # img_135_path = os.path.join(input_dir, name + '_135°.raw')
        # img_135 = preprocess_raw(img_135_path)

        I_alpha = np.concatenate(
            [img_0[:, :, :, None], img_45[:, :, :, None], img_90[:, :, :, None]],
            axis=3
        )  # (H, W, C, 3)
        print(I_alpha.shape)

        # use linear equation to solve I and delta_I
        # use "_hat" to denote the estimated parameters
        # [1/2, -cos(2*alpha)/2, -sin(2*alpha)/2]*[I, I*P*cos(2*theta_parallel), I*P*sin(2*theta_parallel)]^T = I_alpha
        # delta_I = I_perpendicular - I_parallel = I*(1+P)/2 - I*(1-P)/2 = I*P
        coefficient_matrix = np.array(
            [[1 / 2, -np.cos(2 * alpha[0]) / 2, -np.sin(2 * alpha[0]) / 2],
             [1 / 2, -np.cos(2 * alpha[1]) / 2, -np.sin(2 * alpha[1]) / 2],
             [1 / 2, -np.cos(2 * alpha[2]) / 2, -np.sin(2 * alpha[2]) / 2], ], dtype=np.float32
        )  # (3, 3)
        solution = np.linalg.solve(coefficient_matrix[None, None, None, :, :], I_alpha)  # (H, W, C, 3)
        solution = np.float32(solution)
        solution0 = np.clip(solution[:, :, :, 0], a_min=0, a_max=1)
        solution1 = np.clip(solution[:, :, :, 1], a_min=0, a_max=1)
        solution2 = np.clip(solution[:, :, :, 2], a_min=-1, a_max=1)
        I_hat = np.clip(solution0, a_min=0, a_max=1)  # (H, W, C)
        P_hat = np.clip(
            np.sqrt(solution1 ** 2 + solution2 ** 2) / (solution0 + 1e-7), a_min=0, a_max=1
        )  # (H, W, C)
        delta_I_hat = I_hat * P_hat  # (H, W, C)
        I_alpha = np.concatenate([img_0[:, :, :], img_45[:, :, :], img_90[:, :, :]], axis=2)  # (H, W, 3*C)

        np.save(os.path.join(I_alpha_dir, name + '.npy'), I_alpha)
        np.save(os.path.join(I_hat_dir, name + '.npy'), I_hat)
        np.save(os.path.join(delta_I_hat_dir, name + '.npy'), delta_I_hat)
