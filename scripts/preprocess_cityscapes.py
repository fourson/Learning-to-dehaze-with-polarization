import os
import numpy as np
import cv2

# edit these parameters
mode = 'train'
# mode = 'val'
output_subdir_name = 'half_res'
# output_subdir_name = 'quarter_res'
scale = 0.5
# scale = 0.25
image_dir = '../cityscapes_unchanged/leftImg8bit/' + mode  # dir for 'leftImg8bit'
segmentation_dir = '../cityscapes_unchanged/gtFine/' + mode  # 'gtFine'
transmittance_dir = '../cityscapes_unchanged/leftImg8bit_transmittanceDBF/' + mode  # 'transmittanceDBF'
output_dir = '../cityscapes_ours/' + output_subdir_name + '/' + mode


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


image_output_dir = os.path.join(output_dir, 'image')
segmentation_output_dir = os.path.join(output_dir, 'segmentation')
depth_output_dir = os.path.join(output_dir, 'depth')
ensure_dir(image_output_dir)
ensure_dir(segmentation_output_dir)
ensure_dir(depth_output_dir)

beta = 0.01  # use this value of beta for computing the depth
crop = True  # crop the peripheral pixels ("out of ROI" pixels)
crop_num = 32

for subdir_name in os.listdir(image_dir):
    image_subdir = os.path.join(image_dir, subdir_name)
    segmentation_subdir = os.path.join(segmentation_dir, subdir_name)
    transmittance_subdir = os.path.join(transmittance_dir, subdir_name)

    for image_name in os.listdir(image_subdir):
        name = '_'.join(image_name.split('_', 3)[:3])

        segmentation_name = name + '_gtFine_labelIds' + '.png'
        transmittance_name = name + '_leftImg8bit_transmittance_beta_' + str(beta) + '.png'

        image_path = os.path.join(image_subdir, image_name)
        segmentation_path = os.path.join(segmentation_subdir, segmentation_name)
        transmittance_path = os.path.join(transmittance_subdir, transmittance_name)

        print('input paths:')
        print(image_path)
        print(segmentation_path)
        print(transmittance_path)

        image = cv2.imread(image_path)
        segmentation = cv2.imread(segmentation_path)
        transmittance = cv2.imread(transmittance_path)

        if crop:
            image = image[crop_num:-crop_num, crop_num:-crop_num, :]
            segmentation = segmentation[crop_num:-crop_num, crop_num:-crop_num, :]
            transmittance = transmittance[crop_num:-crop_num, crop_num:-crop_num, :]

        H, W, _ = image.shape
        H_new = int(H * scale)
        W_new = int(W * scale)

        image = cv2.resize(image, (W_new, H_new), interpolation=cv2.INTER_NEAREST)
        segmentation = cv2.resize(segmentation, (W_new, H_new), interpolation=cv2.INTER_NEAREST)
        transmittance = cv2.resize(transmittance, (W_new, H_new), interpolation=cv2.INTER_NEAREST)

        # convert the order from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.  # scale to [0, 1]
        # just keep a single channel
        segmentation = segmentation[:, :, 0]
        transmittance = transmittance[:, :, 0]
        transmittance = np.float32(transmittance) / 255.  # scale to [0, 1]
        depth = np.log(transmittance) / (-beta)

        image_output_path = os.path.join(image_output_dir, name + '.npy')
        segmentation_output_path = os.path.join(segmentation_output_dir, name + '.npy')
        depth_output_path = os.path.join(depth_output_dir, name + '.npy')

        print('output paths:')
        print(image_output_path)
        print(segmentation_output_path)
        print(depth_output_path)

        np.save(image_output_path, image)
        np.save(segmentation_output_path, segmentation)
        np.save(depth_output_path, depth)
