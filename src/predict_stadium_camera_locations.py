import cv2
import json
from tabulate import tabulate
import numpy as np
from src.modules.synthetic_camera.main import MainSyntheticCamera
from src.modules.Camera import decompose_homography_to_parameters
import src.scripts.unused.functions as functions
from src.utils import utils


def read_json_point_annotations(filename):
    with open(filename) as file:
        data = json.load(file)
        file.close()

    im_h, im_w = data['imageHeight'], data['imageWidth']
    w_factor, h_factor = 1280 / im_w, 720 / im_h

    point_annotations = {}
    point_annotations['image_height'] = 720
    point_annotations['image_width'] = 1280
    point_annotations['points'] = {}
    for annotation in data['shapes']:
        point_descr = annotation['label']
        point = transform_for_output_resolution(annotation['points'][0], w_factor, h_factor)
        point_annotations['points'][point_descr] = list(point)
    return point_annotations

def transform_for_output_resolution(p, w_f, h_f):
    x, y = p
    return x * w_f, y * h_f

def estimate_camera_location(src_points, annotation):
    im_w, im_h = annotation['image_width'], annotation['image_height']
    annotation_points = [descr for descr in annotation['points']]
    imgpoints = np.array([annotation['points'][descr] for descr in annotation_points], dtype=np.float32).squeeze()
    objpoints = np.array([src_points[descr][:2] for descr in annotation_points], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(objpoints, imgpoints)
    camera_params = decompose_homography_to_parameters(homography, im_w/2, im_h/2)
    x, y, z = camera_params[6:]
    return x, y, z

def get_template_points(camera_descr):
    if camera_descr == 'offside_left':
        return {
            'Big rect. bottom left': template['1. Big rect. left bottom'][0],
            'Big rect. bottom right': template['1. Big rect. left bottom'][-1],
            'Big rect. top right': template['3. Big rect. left top'][-1],
            'Big rect. top left': template['3. Big rect. left top'][0],
            'Small rect. bottom left': template['21. Small rect. left bottom'][0],
            'Small rect. bottom right': template['21. Small rect. left bottom'][-1],
            'Small rect. top right': template['23. Small rect. left top'][-1],
            'Small rect. top left': template['23. Small rect. left top'][0],
            'Top left corner': template['18. Side line left'][-1],
            'Bottom left corner': template['18. Side line left'][0],
            'Circle left top': template['8. Circle left'][-1],
            'Circle left bottom': template['8. Circle left'][0],
            'Circle central top': template['7. Circle central'][16],
            'Circle central bottom': template['7. Circle central'][48],
            'Middle line top': template['16. Middle line'][-1],
            'Middle line bottom': template['16. Middle line'][0],
        }
    elif camera_descr == 'offside_right':
        return {
            'Big rect. bottom left': template['4. Big rect. right bottom'][-1],
            'Big rect. bottom right': template['4. Big rect. right bottom'][0],
            'Big rect. top right': template['6. Big rect. right top'][0],
            'Big rect. top left': template['6. Big rect. right top'][-1],
            'Small rect. bottom left': template['24. Small rect. right bottom'][-1],
            'Small rect. bottom right': template['24. Small rect. right bottom'][0],
            'Small rect. top right': template['26. Small rect. right top'][0],
            'Small rect. top left': template['26. Small rect. right top'][-1],
            'Top right corner': template['19. Side line right'][-1],
            'Bottom right corner': template['19. Side line right'][0],
            'Circle right bottom': template['9. Circle right'][0],
            'Circle right top': template['9. Circle right'][-1],
            'Circle central top': template['7. Circle central'][16],
            'Circle central bottom': template['7. Circle central'][48],
            'Middle line top': template['16. Middle line'][-1],
            'Middle line bottom': template['16. Middle line'][0],
        }
    elif 'behind_right' in camera_descr:
        return {
            'Left corner': template['19. Side line right'][0],
            'Left middle': template['16. Middle line'][0],
            'Right corner': template['19. Side line right'][-1],
            'Right middle': template['16. Middle line'][-1],
            'Circle central left': template['7. Circle central'][48],
            'Circle central right': template['7. Circle central'][16],
            'Circle side left': template['9. Circle right'][0],
            'Circle side right': template['9. Circle right'][-1],
            'Big rect. far left': template['4. Big rect. right bottom'][-1],
            'Big rect. close left': template['4. Big rect. right bottom'][0],
            'Big rect. far right': template['6. Big rect. right top'][-1],
            'Big rect. close right': template['6. Big rect. right top'][0],
            'Small rect. far left': template['24. Small rect. right bottom'][-1],
            'Small rect. close left': template['24. Small rect. right bottom'][0],
            'Small rect. far right': template['26. Small rect. right top'][-1],
            'Small rect. close right': template['26. Small rect. right top'][0],
        }
    elif 'behind_left' in camera_descr:
        return {
            'Left corner': template['18. Side line left'][-1],
            'Left middle': template['16. Middle line'][-1],
            'Right corner': template['18. Side line left'][0],
            'Right middle': template['16. Middle line'][0],
            'Circle central left': template['7. Circle central'][16],
            'Circle central right': template['7. Circle central'][48],
            'Circle side left': template['8. Circle left'][-1],
            'Circle side right': template['8. Circle left'][0],
            'Big rect. far left': template['3. Big rect. left top'][-1],
            'Big rect. close left': template['3. Big rect. left top'][0],
            'Big rect. far right': template['1. Big rect. left bottom'][-1],
            'Big rect. close right': template['1. Big rect. left bottom'][0],
            'Small rect. far left': template['23. Small rect. left top'][-1],
            'Small rect. close left': template['23. Small rect. left top'][0],
            'Small rect. far right': template['21. Small rect. left bottom'][-1],
            'Small rect. close right': template['21. Small rect. left bottom'][0],
        }
    else:
        raise AssertionError('Invalid camera description!')


if __name__ == '__main__':
    camera_descr = 'high_behind_right'
    template = utils.get_court_template()
    filepath = f'{utils.get_dataset_root()}dfva/match_analytics/stadiums/{camera_descr}/'
    master_stadiums = [
        'oaka',
        # 'agrinio2',
        # 'giannena',
        # 'ofi',
        # 'aris'
    ]

    locations = []
    if camera_descr == 'master':
        model_filepath = f"{utils.get_generated_models_path()}main_eff-b0_bs4_lr1e-03_st[5, 10]_res(640, 360)_norm[-1, 1]_@1M_e15_sc0.17.pth"
        main = MainSyntheticCamera(model_filepath)
        for name in master_stadiums:
            line_annotation = functions.read_json_line_annotations(f'{filepath}{name}.json')
            line_annotation = functions.transform_annotation_for_resolution(line_annotation, (1280, 720))
            edge_map = functions.read_edge_map(line_annotation)
            x, y, z = main.estimate_camera_location(edge_map)
            locations.append(
                [f'{name}', round(x, 2), round(y, 2), round(z, 2)])
    else:
        template_points = get_template_points(camera_descr)
        for name in master_stadiums:
            point_annotation = read_json_point_annotations(f'{filepath}{name}.json')
            x, y, z = estimate_camera_location(template_points, point_annotation)
            locations.append(
                [f'{name}', round(x, 2), round(y, 2), round(z, 2)])

    print(camera_descr)
    results = tabulate(locations, headers=['Stadium', 'Pred x', 'Pred y', 'Pred z'])
    print(results)
