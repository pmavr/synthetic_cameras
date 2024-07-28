import scipy.io as sio
import numpy as np
from pathlib import Path
import pickle

import cv2


def get_project_root():
    '''
    :return:  path without slash in the end.
    '''
    path = f'{Path(__file__).parent.parent.parent}/'
    return path

def get_tvcalib_root():
    '''
    :return:  path without slash in the end.
    '''
    path = f'{Path(__file__).parent.parent.parent.parent}/tvcalib/'
    return path


def get_dataset_root():
    path = f'{Path(__file__).parent.parent.parent.parent}/datasets/'
    return path.replace("\\", "/")


def get_dfva_dataset_path():
    return f'{get_dataset_root()}dfva_image_dataset/v2/'


def get_dfva_analytics_dataset_path():
    return f'{get_dataset_root()}dfva_analytics_dataset/'


def get_tswc_dataset_path():
    return f'{get_dataset_root()}TS-WorldCup/'


def get_wc14_dataset_path():
    return f'{get_dataset_root()}world_cup_2014/'

def get_soccernet_dataset_path():
    return f'{get_dataset_root()}soccernet_v2_lines/'

def get_court_template(resolution=1):
    filename = f'{get_project_root()}src/utils/court_template_3d.pkl'
    file = open(filename, 'rb')
    court_lines = pickle.load(file)
    file.close()
    template = {}
    template['lines'] = {}
    template['grid'] = {}
    for line in court_lines:
        if ('Circle' in line) or ('post' in line):
            template['lines'][line] = court_lines[line]
            continue
        template['lines'][line] = interpolate_between_line_endpoints(court_lines[line], resolution)

    court_length_x, court_width_y, _ = template['lines']['20. Side line top'][-1]
    u, v = np.mgrid[0:court_length_x:resolution, 0:court_width_y:resolution]
    template['grid']['points'] = np.stack([u.flatten(), v.flatten()], axis=1)
    return template['lines']


def interpolate_between_line_endpoints(line, res):
    x_values, y_values, z_values = line[:, 0], line[:, 1], line[:, 2]
    if x_values[0] != x_values[-1]:
        x_res = int(np.linalg.norm(x_values) / res)
        x_values = np.linspace(x_values[0], x_values[-1], x_res)
        y_values = np.ones_like(x_values) * y_values.max()
    if y_values[0] != y_values[-1]:
        y_res = int(np.linalg.norm(y_values) / res)
        y_values = np.linspace(y_values[0], y_values[-1], y_res)
        x_values = np.ones_like(y_values) * x_values.max()
    z_values = np.ones_like(x_values) * z_values[0]
    interpolated_line = np.stack([x_values, y_values, z_values], axis=1)
    return interpolated_line


def get_generated_models_path():
    return f'{get_project_root()}src/models/generated_models/'


def get_chen_2019_module_path():
    return f'{get_project_root()}src/modules/chen_2019/'


def get_chu_2022_module_path():
    return f'{get_project_root()}src/modules/chu_2022/'


def get_yolov5_module_path():
    return f'{get_project_root()}src/modules/line_detection/yolov5/'


def get_chen_2019_binary_court():
    return sio.loadmat(f'{get_chen_2019_module_path()}files/worldcup2014.mat')


def get_chen_2019_edge_maps():
    data =  sio.loadmat(f'{get_chen_2019_module_path()}files/testset_feature.mat')
    return data['edge_map']

def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if not isinstance(img_list, list):
        return 'Input is not a list.'

    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]
    else:
        msg_list = [f'{msg}' for msg in msg_list]

    for i in range(len(img_list)):
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv2.destroyWindow(msg)


def tensor2image(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def apply_binary_mask(image, mask):
    """Both input have to have 3 channels."""
    output_image = np.copy(image)
    height, width = image.shape[0], image.shape[1]
    _mask = cv2.resize(mask, (width, height))
    normalized_mask = _mask / 255
    output_image = np.multiply(output_image, normalized_mask)
    return output_image.astype(np.uint8)


def read_text_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines




def gen_template_grid():
    # === set uniform grid ===
    # field_dim_x, field_dim_y = 105.000552, 68.003928 # in meter
    field_dim_x, field_dim_y = 114.83, 74.37  # in yard
    # field_dim_x, field_dim_y = 115, 74 # in yard
    nx, ny = (13, 7)
    x = np.linspace(0, field_dim_x, nx)
    y = np.linspace(0, field_dim_y, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
    uniform_grid = np.concatenate((uniform_grid, np.ones(
        (uniform_grid.shape[0], 1))), axis=1)  # top2bottom, left2right
    # TODO: class label in template, each keypoints is (x, y, c), c is label that starts from 1
    for idx, pts in enumerate(uniform_grid):
        pts[2] = idx + 1  # keypoints label
    return uniform_grid


if __name__ == '__main__':
    from glob import glob
    import os
    path = f"{get_dfva_analytics_dataset_path()}AEK_VS_ARIS_8_2_21/FIRST_HALF/"
    os.chdir(path)
    dirs = os.listdir(os.getcwd())

    old_name = dirs[0]
    start, end = dirs[0].split('_')[-1].split('-')
    start = "".join(start.split(':')[1:])
    end = "".join(end.split(':')[1:])
    new_name = f"event_{start}-{end}"
    os.rename(old_name, new_name)
    # filenames = glob(f"{annotations_path}{dir}/*.npy")
    # filenames = [f.split('/')[-1] for f in filenames]
    # filenames = [f.split('_homography.npy')[0] for f in filenames]


    print('gr')


