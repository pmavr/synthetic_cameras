import cv2
from tqdm import tqdm
import numpy as np
import glob
import json
from collections import defaultdict

from src.modules.synthetic_camera.aux import AuxEccSyntheticCamera
from src.modules.Camera import Camera
from src.utils import utils
from src.utils.evaluation_metrics import Metrics

stadium_dictionary = {
    'AEK': 'oaka',
    'PANE': 'agrinio',
    'ATR': 'peristeri',
    'APO': 'smyrni',
    'PAS': 'giannena'
}

viewpoint_dictionary = [
    'master',
    'offl',
    'offr',
    'hbl',
    'hbr'
]

def get_models():
    models = {}
    path = f"{utils.get_generated_models_path()}dfva/"
    viewpoint_directories = glob.glob(f"{path}*")
    viewpoint_directories = [d.replace('\\', '/').split('/')[-1] for d in viewpoint_directories]
    for d in viewpoint_directories:
        if d not in viewpoint_dictionary:
            continue
        tmp = glob.glob(f"{path}{d}/*.pth")
        image_files = [img.replace('\\', '/').split('/')[-1] for img in tmp]
        model_filenames = [img.split('.')[0] for img in image_files]

        for f in model_filenames:
            model_filename = f'{path}{d}/{f}.pth'
            print(model_filename)
            models[f] = AuxEccSyntheticCamera(model_filename, (720, 576), (1280, 720), iter_num=30, eps_thres=1e-4,
                                              dist_thresh=15)
    return models


def get_dfva_filenames(annotations_path):
    info_file = f"{annotations_path}demo_info_cam_gt.json"
    with open(info_file) as file:
        data = json.load(file)
        file.close()

    filenames = defaultdict(list)

    for f in data:
        camera_descr = data[f]['camera']
        stadium = data[f]['stadium']
        d = {
            'name': f.split('.jpg')[0],
            'stadium': stadium
        }
        filenames[camera_descr].append(d)

    return filenames


def read_json_line_annotations(filename):
    with open(filename) as file:
        data = json.load(file)
        file.close()

    line_annotations = {}
    for annotation in data:
        if 'Goal' in annotation:
            continue
        line_annotations[annotation] = data[annotation]
    return line_annotations


def read_edge_map(line_annotation, color=(255, 255, 255), im_w=1280, im_h=720, thickness=2):
    edge_map = np.zeros((im_h, im_w, 3), dtype=np.uint8)

    for descr in line_annotation:
        num_of_points = len(line_annotation[descr])
        points = line_annotation[descr]
        for i in range(1, num_of_points):
            q1, q2 = points[i - 1], points[i]
            q1 = round(q1['x'] * im_w), round(q1['y'] * im_h)
            q2 = round(q2['x'] * im_w), round(q2['y'] * im_h)
            cv2.line(edge_map, tuple(q1), tuple(q2), color=color, thickness=thickness)
    return edge_map



if __name__ == '__main__':
    template = utils.get_court_template(resolution=2)
    models = get_models()
    data_filepath = f'{utils.get_dfva_dataset_path()}test/'
    image_filepath = f'{data_filepath}images/'
    annotation_filepath = f'{data_filepath}labels_panoramic_sn-2022_format/'
    tvcalib_extremities_filepath = f'{utils.get_tvcalib_root()}data/segment_localization/output/dfva/np8_nc10_r4_md60/'
    pred_extremities_json_path = f'{utils.get_project_root()}demo/extremities/tvcalib/dfva/'
    im_w, im_h = 720, 576
    filenames = get_dfva_filenames(data_filepath)

    # chu_model = ChuEstimator(output_resolution=(720, 576))

    for viewpoint in filenames:
        vp_filenames = filenames[viewpoint]

        for f in tqdm(vp_filenames, leave=False, desc=f'Evaluating for {viewpoint}...'):
            filename, stadium = f['name'], f['stadium']
            im = cv2.imread(f"{image_filepath}{filename}.jpg")

            gt_line_annotation = read_json_line_annotations(f"{annotation_filepath}{filename}.json")
            gt_edge_map = read_edge_map(gt_line_annotation, im_w=im_w, im_h=im_h)

            pred_line_annotation = read_json_line_annotations(
                f"{tvcalib_extremities_filepath}extremities_{filename}.json")
            detected_edge_map = read_edge_map(pred_line_annotation, im_w=im_w, im_h=im_h)
            model_name = f"aux_{stadium}_{viewpoint}"

            if model_name in models:
                # print(f'Predicted camera parameters using {model_name}.')
                #             pred_line_annotation = read_json_line_annotations(f"{pred_extremities_json_path}extremities_{filename}.json")
                #             pred_line_annotation = transform_annotation_for_percent_resolution(pred_line_annotation, im_w=im_w, im_h=im_h)
                #             pred_edge_map = read_edge_map(pred_line_annotation, im_w=im_w, im_h=im_h)
                # pred_edge_map = chu_model.estimate(cv2.resize(im, (1280, 720)))
                # homography = chu_model.last_homography

                pred_edge_map = models[model_name].estimate(detected_edge_map)
                homography = models[model_name].last_homography

                pred_lines = Camera.get_projected_points(template, homography,
                                                         target_im_res=(im_w, im_h), no_3d_points=False)
                pred_lines = Metrics.remove_lines_that_are_off_image(pred_lines, im_h, im_w)
                pred_extremities_json = Metrics.convert_to_extremities(pred_lines)

                extremities_file = f"{pred_extremities_json_path}extremities_{filename}.json"
            #     with open(extremities_file, "w") as file:
            #         json.dump(pred_extremities_json, file, indent=4)
            #         # np.save(f'{pred_extremities_json_path}{filename}', homography)
            #
            # continue

            output_frame = cv2.addWeighted(src1=im,
                                           src2=pred_edge_map * np.array([0, 0, 1], dtype=np.uint8),
                                           alpha=.95, beta=.9, gamma=0.)
            output_frame = cv2.addWeighted(src1=output_frame,
                                           src2=gt_edge_map * np.array([0, 1, 0], dtype=np.uint8),
                                           alpha=.95, beta=.9, gamma=0.)
            utils.show_image([output_frame], [f"{filename} - {stadium} - {viewpoint}"])
        # print('gr')