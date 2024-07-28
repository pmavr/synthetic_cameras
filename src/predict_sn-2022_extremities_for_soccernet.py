import glob
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
from src.modules.synthetic_camera.aux import AuxSyntheticCamera, AuxEccSyntheticCamera
from src.modules.Camera import decompose_homography_to_parameters, Camera
from src.utils.evaluation_metrics import Metrics
from src.utils import utils




stadium_dictionary = {
    ' Atalanta': 'atalanta',
    ' Barcelona': 'barcelona',
    ' Chelsea': 'chelsea',
    ' Shakhtar Donetsk': 'donetsk',
    ' Dortmund': 'dortmund',
    ' Hamburger SV': 'hamburger',
    ' Hoffenheim': 'hoffenheim',
    ' Inter': 'inter',
    ' Juventus': 'juventus',
    ' Lazio': 'lazio',
    ' RB Leipzig': 'leipzig',
    ' Levante': 'levante',
    ' 1. FSV Mainz 05': 'mainz',
    ' Malaga': 'malaga',
    ' Manchester City': 'mcity',
    ' AC Milan': 'milan',
    ' Bayern Munich': 'munich',
    ' Manchester United': 'mutd',
    ' Napoli': 'napoli',
    ' Norwich': 'norwich',
    ' Las Palmas': 'palmas',
    ' Zenit Petersburg': 'petersburg',
    ' FC Porto': 'porto',
    ' Real Madrid': 'real',
    ' AS Roma': 'roma',
    ' Southampton': 'southampton',
    ' Stoke City': 'stoke',
    ' West Brom': 'west',
    ' Liverpool': 'liverpool',
    ' Sampdoria': 'sampdoria',
    ' Paris SG': 'paris'
}

viewpoint_dictionary = {
    'Main camera center': 'master',
    # 'Main camera left': 'offl',
    # 'Main camera left low': 'offll',
    # 'Main camera right': 'offr',
    # 'Main camera right 1': 'offr1',
    # 'Main camera right 2': 'offr2',
    # 'High behind left': 'hbl',
    # 'High behind right': 'hbr',
    # 'High behind right 1': 'hbr1',
    # 'High behind right 2': 'hbr2',
    # 'High behind right 3': 'hbr3',
    # 'Low behind left': 'lbl'
}


def get_models():
    models = {}
    path = f"{utils.get_generated_models_path()}soccernet/"
    viewpoint_directories = glob.glob(f"{path}*")
    viewpoint_directories = [d.replace('\\', '/').split('/')[-1] for d in viewpoint_directories]
    for d in viewpoint_directories:
        if d not in [item for key, item in viewpoint_dictionary.items()]:
            continue
        tmp = glob.glob(f"{path}{d}/*.pth")
        image_files = [img.replace('\\', '/').split('/')[-1] for img in tmp]
        model_filenames = [img.split('.')[0] for img in image_files]

        for f in model_filenames:
            model_filename = f'{path}{d}/{f}.pth'
            print(model_filename)
            models[f] =  AuxEccSyntheticCamera(model_filename, (960, 540), (1280, 720), iter_num=30, eps_thres=1e-4,
                                              dist_thresh=15)
    return models


def get_soccernet_filenames(annotations_path):
    match_info_cam = f"{annotations_path}match_info_cam_gt.json"
    with open(match_info_cam) as file:
        data = json.load(file)
        file.close()

    filenames = defaultdict(list)

    for f in data:
        camera_descr = data[f]['camera']
        stadium = data[f]['match'].split(' - ')[0]
        d = {
            'name': f.split('.')[0],
            'stadium': stadium
        }
        filenames[camera_descr].append(d)

    filenames = {vp:filenames[vp] for vp in filenames if vp in viewpoint_dictionary}
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


def transform_annotation_for_percent_resolution(line_annotation, im_w, im_h):
    for descr in line_annotation:
        points = line_annotation[descr]
        line_annotation[descr] = [{'x':p['x']/im_w, 'y':p['y']/im_h} for p in points]
    return line_annotation


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
    annotation_filepath = f'{utils.get_soccernet_dataset_path()}test/'
    # tvcalib_extremities_filepath = f'{utils.get_project_root()}demo/line_detect/gt_edge/soccernet-2022_extremities/'
    tvcalib_extremities_filepath = f'{utils.get_tvcalib_root()}data/segment_localization/output/soccernet/np8_nc10_r4_md60/test/'
    pred_extremities_json_path = f'{utils.get_project_root()}demo/extremities/gt_edge/soccernet-2022_master/'
    im_w, im_h = 960, 540
    filenames = get_soccernet_filenames(annotation_filepath)

    for viewpoint in filenames:
        vp_filenames = filenames[viewpoint]

        for f in tqdm(vp_filenames, leave=False, desc=f'Evaluating for {viewpoint}...'):
            filename, stadium = f['name'], f['stadium']
            im = cv2.imread(f"{annotation_filepath}{filename}.jpg")

            gt_line_annotation = read_json_line_annotations(f"{annotation_filepath}{filename}.json")
            gt_edge_map = read_edge_map(gt_line_annotation, im_w=im_w, im_h=im_h)

            pred_line_annotation = read_json_line_annotations(f"{tvcalib_extremities_filepath}extremities_{filename}.json")
            detected_edge_map = read_edge_map(pred_line_annotation, im_w=im_w, im_h=im_h)
            stad = stadium_dictionary[stadium]
            vp = viewpoint_dictionary[viewpoint]
            model_name = f"aux_{stad}_{vp}"

            if model_name in models:
                # print(f'Predicted camera parameters using {model_name}.')
                #             pred_line_annotation = read_json_line_annotations(f"{pred_extremities_json_path}extremities_{filename}.json")
                #             pred_line_annotation = transform_annotation_for_percent_resolution(pred_line_annotation, im_w=im_w, im_h=im_h)
                #             pred_edge_map = read_edge_map(pred_line_annotation, im_w=im_w, im_h=im_h)

                pred_edge_map = models[model_name].estimate(gt_edge_map)
                homography = models[model_name].last_homography
                pred_lines = Camera.get_projected_points(template, homography,
                                                         target_im_res=(im_w, im_h), no_3d_points=False)
                pred_lines = Metrics.remove_lines_that_are_off_image(pred_lines, im_h, im_w)
                pred_extremities_json = Metrics.convert_to_extremities(pred_lines)

            #     extremities_file = f"{pred_extremities_json_path}extremities_{filename}.json"
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
            utils.show_image([output_frame], [f"{filename} - {stad} - {vp}"])
        # print('gr')
