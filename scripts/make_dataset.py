import os
import random
import argparse
import numpy as np
import scipy.ndimage
from tqdm import tqdm


class RandomParameters:
    # for segmentation
    seg_cls_number = 34

    sky_cls_id = 23  # start from 0

    def __init__(self, segmentation, channel, sv_A_infinity, sv_beta):
        self.H = segmentation.shape[0]
        self.W = segmentation.shape[1]
        self.C = channel
        self.sv_A_infinity = sv_A_infinity
        self.sv_beta = sv_beta

        self.A_infinity_base = np.random.uniform(0.85, 0.95)
        self.A_infinity_fluctuation = self.A_infinity_base * 0.05

        # self.beta_choices = np.arange(0.005, 0.01, 0.02)  # if you want to use fixed value
        # self.beta_base = np.random.choice(self.beta_choices)
        self.beta_base = np.random.uniform(0.01, 0.02)
        self.beta_fluctuation = self.beta_base * 0.05

        # polarization
        self.P_A_base = np.random.uniform(0.05, 0.4)
        self.P_A_fluctuation = self.P_A_base * 0.02

        self.P_T_coarse_values = np.random.uniform(0.025, 0.2, (self.seg_cls_number,))
        self.P_T_coarse_mean = self.P_T_coarse_values[segmentation]  # (H, W)
        self.P_T_coarse_std = self.P_T_coarse_mean * 0.01
        self.P_T_coarse = np.random.normal(self.P_T_coarse_mean, self.P_T_coarse_std)  # (H, W)
        self.P_T_coarse[segmentation == self.sky_cls_id] = 0  # set sky regions to zero
        self.P_T_coarse = scipy.ndimage.gaussian_filter(self.P_T_coarse, sigma=3)
        self.P_T_base = np.broadcast_to(np.expand_dims(self.P_T_coarse, 2), (self.H, self.W, self.C))  # (H, W, C)
        self.P_T_fluctuation = self.P_T_base * 0.02

        self.theta_parallel_mean = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.theta_parallel_std = np.abs(self.theta_parallel_mean) * 0.02

    def generate(self):
        parameters = dict()
        if self.sv_A_infinity:
            parameters['A_infinity'] = np.sort(
                np.float32(
                    np.random.uniform(-self.A_infinity_fluctuation, self.A_infinity_fluctuation,
                                      (self.H, self.W, self.C)) + self.A_infinity_base
                )
            )  # (H, W, C)  A_infinity_r < A_infinity_g < A_infinity_b if RGB
        else:
            parameters['A_infinity'] = np.sort(
                np.float32(
                    np.random.uniform(-self.A_infinity_fluctuation, self.A_infinity_fluctuation,
                                      (self.C,)) + self.A_infinity_base
                )
            )  # (C,)  A_infinity_r < A_infinity_g < A_infinity_b if RGB
            parameters['A_infinity'] *= np.ones((self.H, self.W, self.C))  # convert to (H, W, C)
        if self.sv_beta:
            parameters['beta'] = np.sort(
                np.float32(
                    np.random.uniform(-self.beta_fluctuation, self.beta_fluctuation,
                                      (self.H, self.W, self.C)) + self.beta_base
                )
            )  # (H, W, C)  beta_r < beta_g < beta_b if RGB
        else:
            parameters['beta'] = np.sort(
                np.float32(
                    np.random.uniform(-self.beta_fluctuation, self.beta_fluctuation, (self.C,)) + self.beta_base
                )
            )  # (C,)  beta_r < beta_g < beta_b if RGB
            parameters['beta'] *= np.ones((self.H, self.W, self.C))  # convert to (H, W, C)
        parameters['P_A'] = np.clip(
            -np.sort(
                -np.float32(
                    np.random.uniform(-self.P_A_fluctuation, self.P_A_fluctuation, (self.C,)) + self.P_A_base
                )
            ),
            a_min=0,
            a_max=1
        )  # (C,)  P_A_r > P_A_g > P_A_b if RGB

        parameters['P_T'] = np.clip(
            -np.sort(
                -np.float32(
                    np.random.uniform(-self.P_T_fluctuation, self.P_T_fluctuation,
                                      (self.H, self.W, self.C)) + self.P_T_base
                ), axis=2
            ),
            a_min=0,
            a_max=1
        )  # (H, W, C)  P_T_r > P_T_g > P_T_b if RGB
        parameters['theta_parallel'] = np.clip(
            np.float32(
                np.random.normal(self.theta_parallel_mean, self.theta_parallel_std, (self.H, self.W, self.C))
            ),
            a_min=-np.pi / 4,
            a_max=np.pi / 4
        )  # (C,)
        return parameters


class DataItem:
    # for polarizer
    alpha = np.array([0, np.pi / 4, np.pi / 2], dtype=np.float32)  # (3,)

    def __init__(self, R, Z, A_infinity, beta, P_A, P_T, theta_parallel):
        self.valid = True  # for validity check

        # radiance
        self.R = R  # (H, W, C)
        # depth
        self.Z = Z  # (H, W, 1)
        # airlight radiance corresponding to an object at an infinite distance
        self.A_infinity = A_infinity  # (H, W, C)
        # scattering coefficient
        self.beta = beta  # (H, W, C)
        # airlight degree of polarization, defined by: P_A = (A_perpendicular - A_parallel)/A
        self.P_A = P_A  # (C,)
        # transmission light degree of polarization, defined by: P_T = (T_perpendicular - T_parallel)/T
        self.P_T = P_T  # (H, W, C)
        # orientation of the polarizer for best transmission of the component parallel to the plane of incidence
        self.theta_parallel = theta_parallel  # (H, W, C)

        # transmittance
        self.t = np.exp(-self.beta * self.Z, dtype=np.float32)  # (H, W, C)

        # transmission light
        self.T = self.R * self.t  # (H, W, C)
        # airlight
        self.A = self.A_infinity * (1 - self.t)  # (H, W, C)
        # total light
        self.I = self.T + self.A  # (H, W, C)
        # the PD (polarization difference) image, calculated by: delta_I = P*I = A*P_A + T*P_T = delta_A + delta_T
        self.delta_I = self.A * self.P_A + self.T * self.P_T  # (H, W, C)
        # total light degree of polarization, defined by: P = (I_perpendicular - I_parallel)/I
        self.P = self.delta_I / (self.I + 1e-7)  # (H, W, C)

        # generate three polarized images
        cos_term = np.float32(np.cos(2 * (self.alpha[:, None, None, None] - self.theta_parallel)))  # (3, H, W, C)
        self.I_alpha = self.I / 2 - self.delta_I * cos_term / 2  # (3, H, W, C)
        # add noise for polarized images
        self.I_alpha = np.float32(np.random.normal(self.I_alpha, self.I_alpha * 0.02))
        self.I_alpha = np.clip(self.I_alpha, a_min=0, a_max=1)

        # use linear equation to solve I and delta_I
        # use "_hat" to denote the estimated parameters
        # [1/2, -cos(2*alpha)/2, -sin(2*alpha)/2]*[I, I*P*cos(2*theta_parallel), I*P*sin(2*theta_parallel)]^T = I_alpha
        # delta_I = I_perpendicular - I_parallel = I*(1+P)/2 - I*(1-P)/2 = I*P
        coefficient_matrix = np.array(
            [[1 / 2, -np.cos(2 * self.alpha[0]) / 2, -np.sin(2 * self.alpha[0]) / 2],
             [1 / 2, -np.cos(2 * self.alpha[1]) / 2, -np.sin(2 * self.alpha[1]) / 2],
             [1 / 2, -np.cos(2 * self.alpha[2]) / 2, -np.sin(2 * self.alpha[2]) / 2], ], dtype=np.float32
        )  # (3, 3)
        solution = np.linalg.solve(
            coefficient_matrix[None, None, None, :, :], np.transpose(self.I_alpha, (1, 2, 3, 0))
        )  # (H, W, C, 3)
        solution = np.float32(solution)
        # solution0 = solution[:, :, :, 0] # something weird would happen...
        # solution1 = solution[:, :, :, 1]
        # solution2 = solution[:, :, :, 2]
        solution0 = np.clip(solution[:, :, :, 0], a_min=0, a_max=1)
        solution1 = np.clip(solution[:, :, :, 1], a_min=0, a_max=1)
        solution2 = np.clip(solution[:, :, :, 2], a_min=-1, a_max=1)

        self.I_hat = np.clip(solution0, a_min=0, a_max=1)  # (H, W, C)
        self.P_hat = np.clip(
            np.sqrt(solution1 ** 2 + solution2 ** 2) / (solution0 + 1e-7), a_min=0, a_max=1
        )  # (H, W, C)
        self.delta_I_hat = self.I_hat * self.P_hat  # (H, W, C)

        # convert I_alpha to # (H, W, 3*C)
        self.I_alpha = np.concatenate([self.I_alpha[0], self.I_alpha[1], self.I_alpha[2]], axis=2)  # (H, W, 3*C)

        # sanity check
        # if np.isnan(self.I_hat).any() or np.isnan(self.P_hat).any():
        #     self.valid = False
        #     return


class DatasetMaker:
    def __init__(self, image_dir, depth_dir, segmentation_dir, output_dir, number_of_samples, size=None, mode='train',
                 prefix='', crop_mode='random', sv_A_infinity=1, sv_beta=1):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.segmentation_dir = segmentation_dir
        self.output_dir = output_dir
        self.number_of_samples = number_of_samples
        if size:
            self.do_crop = True
            self.H, self.W = size[0], size[1]
            self.crop_mode = crop_mode
        else:
            self.do_crop = False
        self.C = None

        if mode == 'train':
            self.output_subdir_names = ['I_alpha', 'I_hat', 'I', 'delta_I_hat', 'delta_I', 'A_infinity', 'P_A', 'P_T',
                                        'T', 'R']
        elif mode == 'val':
            self.output_subdir_names = ['I_alpha', 'I_hat', 'I', 'delta_I_hat', 'delta_I', 'A_infinity', 'P_A', 'P_T',
                                        'T', 'R', 'segmentation', 'P', 'P_hat']
        else:
            raise Exception('mode not defined!')

        self.prefix = prefix
        self.sv_A_infinity = sv_A_infinity
        self.sv_beta = sv_beta

    def make(self):
        for file_name in tqdm(os.listdir(self.image_dir)[:self.number_of_samples], ascii=True):
            print(file_name)
            image_path = os.path.join(self.image_dir, file_name)
            depth_path = os.path.join(self.depth_dir, file_name)
            segmentation_path = os.path.join(self.segmentation_dir, file_name)
            image = np.load(image_path).astype(np.float32)  # (H, W, C)
            if len(image.shape) == 2:
                # if the image is a grayscale one in (H, W) shape, we should turn it to (H, W, 1)
                image = np.expand_dims(image, 2)
                self.C = 1
            else:
                # if the image is in (H, W, C) shape, just keep it
                self.C = image.shape[2]

            depth = np.expand_dims(np.load(depth_path).astype(np.float32), 2)  # (H, W, 1)
            segmentation = np.load(segmentation_path)  # (H, W)

            name = self.prefix + file_name.split('.')[0]
            H, W, _ = image.shape
            if self.do_crop:
                if self.H > H or self.W > W:
                    print('crop_size too large, discard')
                    continue

                if self.crop_mode == 'top_left':
                    # cut top_left
                    image = image[0:self.H, 0:self.W, :]
                    depth = depth[0:self.H, 0:self.W, :]
                    segmentation = segmentation[0:self.H, 0:self.W]
                elif self.crop_mode == 'top_right':
                    # cut top_right
                    image = image[0:self.H, W - self.W:W, :]
                    depth = depth[0:self.H, W - self.W:W, :]
                    segmentation = segmentation[0:self.H, W - self.W:W]
                else:
                    # random cut
                    h_offset = random.randint(0, H - self.H)
                    w_offset = random.randint(0, W - self.W)
                    image = image[h_offset:h_offset + self.H, w_offset:w_offset + self.W, :]
                    depth = depth[h_offset:h_offset + self.H, w_offset:w_offset + self.W, :]
                    segmentation = segmentation[h_offset:h_offset + self.H, w_offset:w_offset + self.W]

                print('shape: crop', (H, W), 'to', (self.H, self.W))
            else:
                print('shape: ', (H, W))

            data_item = DataItem(
                image, depth, **RandomParameters(segmentation, self.C, self.sv_A_infinity, self.sv_beta).generate()
            )
            if not data_item.valid:
                print('invalid... just leave it out...')
                continue

            for output_subdir_name in self.output_subdir_names:
                output_subdir = os.path.join(self.output_dir, output_subdir_name)
                if not os.path.exists(output_subdir):
                    os.mkdir(output_subdir)

                output_path = os.path.join(output_subdir, name + '.npy')
                if output_subdir_name == 'segmentation':
                    value = segmentation
                else:
                    value = getattr(data_item, output_subdir_name)
                np.save(output_path, value)
                print(output_subdir_name, value.max(), value.min(), value.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, type=str, help='base dir for original data')
    parser.add_argument('--subdir', required=True, type=str, choices=['half_res', 'quarter_res'], help='subdir name')
    parser.add_argument('--mode', required=True, type=str, choices=['train', 'val'], help='mode')
    parser.add_argument('--number_of_samples', required=True, type=int, help='number of samples to make')
    parser.add_argument('--size', nargs='*', type=int, help='image size, None for keeping unchanged')
    parser.add_argument('--prefix', required=True, type=str, help='prefix of the output file name')
    parser.add_argument('--output_dir', required=True, type=str, help='output dir')
    parser.add_argument('--crop_mode', required=True, type=str, choices=['top_left', 'top_right', 'random'],
                        help='crop mode')
    parser.add_argument('--sv_A_infinity', default=1, type=int, help='spatially variant A_infinity')
    parser.add_argument('--sv_beta', default=1, type=int, help='spatially variant beta')
    args = parser.parse_args()

    temp_dir = os.path.join(args.base_dir, args.subdir, args.mode)
    image_dir = os.path.join(temp_dir, 'image')
    depth_dir = os.path.join(temp_dir, 'depth')
    segmentation_dir = os.path.join(temp_dir, 'segmentation')
    output_dir = os.path.join(args.output_dir, args.mode)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dataset_maker = DatasetMaker(
        image_dir,
        depth_dir,
        segmentation_dir,
        output_dir,
        args.number_of_samples,
        args.size,
        args.mode,
        args.prefix,
        args.crop_mode,
        args.sv_A_infinity,
        args.sv_beta
    )
    dataset_maker.make()
