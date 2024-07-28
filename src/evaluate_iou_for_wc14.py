import cv2
from time import time
from tqdm import tqdm
import numpy as np
import json

from src.modules.TopViewer import TopViewer
from src.modules.synthetic_camera.aux import AuxEccSyntheticCamera
from src.modules.chu_2022.ChuEstimator import ChuEstimator
from src.modules.chen_2019.ChenEstimator import ChenEstimator
from src.modules.Camera import Camera
from src.utils.evaluation_metrics import Metrics
from src.utils import utils


def read_homography_file(filename):
    def convert_homography_from_meter2yard(h):
        meter2yard = 1.0936133
        m1 = np.array([[-meter2yard, 0, 0],
                       [0, -meter2yard, 0],
                       [0, 0, 1]])
        h = np.linalg.inv(h)
        h = h @ m1
        m2 = np.asarray([[1, 0, 0],
                         [0, -1, Camera.court_width_y],
                         [0, 0, -1]])
        h = h @ m2
        return h

    with open(filename, 'r') as f:
        mat = [[float(num) for num in line.split(' ') if num != ''] for line in f]
    h = np.array(mat)
    h = convert_homography_from_meter2yard(h)
    return h


if __name__ == '__main__':
    import pickle

    top_viewer = TopViewer(offset=5)
    stadiums = pickle.load(open(f'{utils.get_wc14_dataset_path()}labels/test/cam_info.pkl', 'rb'))
    template = utils.get_court_template(resolution=2)
    calib_method = 'tvcalib'
    seg_method = 'tvcalib'
    images_filepath = f'{utils.get_wc14_dataset_path()}images/test/'
    annotations_filepath = f'{utils.get_wc14_dataset_path()}labels/test/'
    edge_map_filepath = f'{utils.get_project_root()}demo/line_detect/{seg_method}/wc14_4_8/'
    pred_extremities_json_path = f'{utils.get_project_root()}demo/extremities/{calib_method}/wc14_np8_nc10_md60/'
    im_w, im_h = 1280, 720
    scores = {
        'whole_iou': [],
        'partial_iou': []
    }
    for d in stadiums:

        model_filename = f'{utils.get_generated_models_path()}world_cup/aux_v3_{d}_master.pth'
        # model = ChuEstimator(output_resolution=(1280, 720))
        # model = ChenEstimator(output_resolution=(1280, 720))
        # model = AuxEccSyntheticCamera(model_filename, output_im_res=(im_w, im_h), refine_res=(1280, 720),
        #                               iter_num=30,
        #                               eps_thres=1e-4,
        #                               dist_thresh=15)
        # model = AuxEccSyntheticCamera(model_filename, output_im_res=(1280, 720), refine_res=(883, 496), iter_num=1,
        #                                   eps_thres=1e-4,
        #                                   dist_thresh=50)
        # model = AuxSyntheticCamera(model_filename, output_im_res=(1280, 720))

        filenames = stadiums[d]

        for f in tqdm(filenames, leave=False, desc='Evaluating...'):
            im = cv2.imread(f'{images_filepath}{f}.jpg')
            im = cv2.resize(im, (im_w, im_h))
            gt_h = read_homography_file(f'{annotations_filepath}{f}.homographyMatrix')

            gt_edge_map = Camera.to_edge_map_from_h(template, gt_h, im_h, im_w)

            # seg_edge_map = cv2.imread(f'{edge_map_filepath}{f}.jpg')
            # seg_edge_map = seg_edge_map * np.array([0, 1, 0], dtype=np.uint8)
            # pred_edge_map = model.estimate(seg_edge_map)
            # pred_h = model.last_homography[:, [0, 1, 3]]
            
            pred_h = np.load(f'{pred_extremities_json_path}{f}.npy')
            pred_h = pred_h[:, [0, 1, 3]]

            whole_iou = Metrics.whole_iou(gt_h=gt_h, pred_h=pred_h)
            partial_iou = Metrics.partial_iou(gt_h=gt_h, pred_h=pred_h)

            scores['whole_iou'].append(whole_iou)
            scores['partial_iou'].append(partial_iou)

            # continue

            # im = cv2.resize(im, (im_w, im_h))
            # gt_edge_map = cv2.resize(gt_edge_map, (im_w, im_h))
            # pred_edge_map = cv2.resize(pred_edge_map, (im_w, im_h))
            # output_frame = cv2.addWeighted(src1=im,
            #                       src2=pred_edge_map * np.array([1, 0, 1], dtype=np.uint8),
            #                       alpha=.9, beta=1., gamma=0.)
            # output_frame = cv2.addWeighted(src1=output_frame,
            #                       src2=gt_edge_map * np.array([0, 1, 0], dtype=np.uint8),
            #                       alpha=.9, beta=1., gamma=0.)

            proj_pts = Camera.get_projected_points(template, pred_h, no_3d_points=True)
            gt_top_view = top_viewer.draw_projected_points_on_top_view(proj_pts, homography=pred_h, color=(0, 255, 0))
            pred_top_view = top_viewer.draw_projected_points_on_top_view(proj_pts, homography=gt_h, color=(0, 0, 255))

            demo_frame = cv2.addWeighted(src1=gt_top_view,
                                         src2=pred_top_view,
                                         alpha=.95, beta=.9, gamma=0.)

            demo_frame = top_viewer.project_field_of_view_on_top_view(gt_h, demo_frame)

            # cv2.putText(output_frame, f'FPS: {1 / end:.2f}', (20, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            cv2.putText(demo_frame, f'Part. IOU: {partial_iou:.3f}', (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            cv2.putText(demo_frame, f'Whole IOU: {whole_iou:.3f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)

            # utils.show_image([output_frame, demo_frame], [f'rgb_{f}.jpg', f'top_{f}.jpg'])
            utils.show_image([demo_frame], [f'top_{f}.jpg'])
            print('gr')


    print(f"Whole IOU: {np.mean(scores['whole_iou']):.3f}\nPartial IOU: {np.mean(scores['partial_iou']):.3f}")
