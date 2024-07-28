import numpy as np
import cv2
import math
import src.utils.utils as utils


class Camera:
    court_mid_length_x = 52.500276  # meters
    court_mid_width_y = 34.001964
    court_length_x = 105.000552
    court_width_y = 68.003928
    x_shift = 0  # -100
    y_shift = 0  # 75

    def __init__(self, camera_params):
        self.image_center_x = camera_params[0]
        self.image_center_y = camera_params[1]
        self.focal_length = camera_params[2]
        self.tilt_angle = camera_params[3]
        self.pan_angle = camera_params[4]
        self.roll_angle = camera_params[5]
        self.camera_center_x = camera_params[6]
        self.camera_center_y = camera_params[7]
        self.camera_center_z = camera_params[8]
        self.camera_center = camera_params[6:9]
        self.base_rotation = (
            self.rotate_y_axis(0)
            @ self.rotate_z_axis(self.roll_angle)
            @ self.rotate_x_axis(-90)
        )
        pan_tilt_rotation = self.pan_y_tilt_x(self.pan_angle, self.tilt_angle)
        self.rotation_matrix = pan_tilt_rotation @ self.base_rotation
        self.image_width = int(2 * self.image_center_x)
        self.image_height = int(2 * self.image_center_y)

    def calibration_matrix(self):
        return np.array(
            [
                [self.focal_length, 0, self.image_center_x],
                [0, self.focal_length, self.image_center_y],
                [0, 0, 1],
            ]
        )

    def homography(self):
        P = self.projection_matrix()
        return P

    def projection_matrix(self):
        P = np.eye(3, 4)
        P[:, 3] = -1 * self.camera_center
        K = self.calibration_matrix()
        R = self.rotation_matrix
        return K @ R @ P

    @staticmethod
    def rotate_x_axis(angle):
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_y_axis(angle):
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_z_axis(angle):
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        r = np.transpose(r)
        return r

    def pan_y_tilt_x(self, pan, tilt):
        r_tilt = self.rotate_x_axis(tilt)
        r_pan = self.rotate_y_axis(pan)
        m = r_tilt @ r_pan
        return m

    @staticmethod
    def project_point_on_topview(point, h, f_w, f_h):
        x, y = point
        w = 1.0
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w

        scale = np.array([[f_w, 0, 0], [0, f_h, 0], [0, 0, 1]])
        shift = np.array(
            [
                [1, 0, 0 + Camera.x_shift],
                [0, -1, Camera.court_width_y / f_h + Camera.y_shift],
                [0, 0, 1],
            ]
        )

        h = h @ scale
        h = h @ shift
        inv_h = np.linalg.inv(h)
        q = inv_h @ p

        assert q[2] != 0.0
        projected_x = q[0] / q[2]
        projected_y = q[1] / q[2]
        return [projected_x, projected_y]

    @staticmethod
    def project_2d_point_on_frame(point, h):
        x, y, z = point
        p = np.array([x, y, 1.0])
        q = h @ p

        assert q[2] != 0.0
        projected_x = q[0] / q[2]
        projected_y = q[1] / q[2]
        return [projected_x, projected_y]

    @staticmethod
    def project_3d_point_on_frame(point, h):
        x, y, z = point
        p = np.array([x, y, z, 1.0])
        q = h @ p

        assert q[2] != 0.0
        projected_x = q[0] / q[2]
        projected_y = q[1] / q[2]
        return [projected_x, projected_y]

    @staticmethod
    def distance_from_camera(camera_center_z, tilt_angle):
        return camera_center_z / np.cos(np.radians(90 - tilt_angle)) * (-1)

    @staticmethod
    def coords_at_distance(camera_center_y, camera_center_z, tilt_angle, pan_angle):
        straight_dist = camera_center_z * np.tan(np.radians(90 - tilt_angle))
        x_coord = (
            -straight_dist * np.cos(np.radians(90 - pan_angle))
            + Camera.court_mid_length_x
        )
        y_coord = -straight_dist * np.sin(np.radians(90 - pan_angle)) + camera_center_y
        return x_coord, y_coord

    def to_edge_map(self, court_template, color=(255, 255, 255), line_width=2):
        homography = self.homography()
        edge_map = self.to_edge_map_from_h(
            court_template,
            homography,
            self.image_height,
            self.image_width,
            color=color,
            line_width=line_width,
            no_3d_points=False,
        )
        return edge_map

    @staticmethod
    def to_edge_map_from_h(
        court_template,
        homography,
        image_height,
        image_width,
        color=(255, 255, 255),
        line_width=2,
        no_goal_posts=True,
        no_3d_points=True,
    ):
        edge_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        projected_points = Camera.get_projected_points(
            court_template,
            homography,
            no_3d_points,
            target_im_res=(image_width, image_height),
        )
        edge_map = Camera.draw_lines_from_points(
            edge_map,
            projected_points,
            image_height,
            image_width,
            color,
            line_width,
            no_goal_posts,
        )
        return edge_map

    def orientation(self):
        homography = self.homography()
        upper_right_corner = Camera.project_3d_point_on_frame(
            (Camera.court_length_x, Camera.court_width_y, 0.0), homography
        )[0]
        upper_left_corner = Camera.project_3d_point_on_frame(
            (0.0, Camera.court_width_y, 0.0), homography
        )[0]
        lower_right_corner = Camera.project_3d_point_on_frame(
            (Camera.court_length_x, 0.0, 0.0), homography
        )[0]
        lower_left_corver = Camera.project_3d_point_on_frame(
            (0.0, 0.0, 0.0), homography
        )[0]

        if self.image_center_x in range(int(lower_left_corver), int(upper_left_corner)):
            return 1
        elif self.image_center_x in range(
            int(upper_left_corner), int(upper_right_corner)
        ):
            return 0
        elif self.image_center_x in range(
            int(upper_right_corner), int(lower_right_corner)
        ):
            return 2
        else:
            return -1

    @staticmethod
    def get_projected_points(
        court_template, homography, no_3d_points=True, target_im_res=(1280, 720)
    ):
        w_factor, h_factor = target_im_res[0] / 1280, target_im_res[1] / 720
        project_point_on_frame = (
            Camera.project_2d_point_on_frame
            if no_3d_points
            else Camera.project_3d_point_on_frame
        )
        projected_points = {}
        for line in court_template:
            line_pts = court_template[line]
            proj_pts = []
            for i in range(len(line_pts)):
                pt = line_pts[i]
                proj_pt = project_point_on_frame(pt, homography)
                proj_pt = [proj_pt[0] * w_factor, proj_pt[1] * h_factor]
                proj_pts.append(proj_pt)
            projected_points[line] = np.array(proj_pts)

        return projected_points

    @staticmethod
    def point_to_int(p):
        return [int(_) for _ in p]

    @staticmethod
    def draw_lines_from_points(
        edge_map,
        points,
        image_height,
        image_width,
        color=(255, 255, 255),
        line_width=2,
        no_goal_posts=True,
    ):
        def _is_off_image_point(point):
            x, y = point
            return x < 0 or y < 0 or x > image_width or y > image_height

        output_img = np.copy(edge_map)

        for group in points:
            if no_goal_posts and ("Goal" in group):
                continue

            for i in range(len(points[group])):
                startpoint, endpoint = points[group][i - 1], points[group][i]
                if _is_off_image_point(startpoint) and _is_off_image_point(endpoint):
                    continue
                startpoint = Camera.point_to_int(startpoint)
                endpoint = Camera.point_to_int(endpoint)
                cv2.line(
                    output_img,
                    tuple(startpoint),
                    tuple(endpoint),
                    color=color,
                    thickness=line_width,
                )
        return output_img


def decompose_homography_to_parameters(homography, u, v):
    result = from_homography(homography, u, v)
    if not result:
        return None
    K, R, CC = result
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(R)
    pan, tilt, roll = np.degrees(pan) * (-1), np.degrees(tilt) + 90, np.degrees(roll)
    return np.array([u, v, K[0, 0], tilt, pan, roll, CC[0], CC[1], CC[2]])


def from_homography(homography, u, v):
    """
    This method initializes the essential camera parameters from the homography between the world plane of the pitch
    and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
    Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
    same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
    computer vision, p196).
    :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
     and its image.
    """
    success, calibration = estimate_calibration_matrix_from_plane_homography(
        homography, u, v
    )
    if not success:
        return False

    hprim = np.linalg.inv(calibration) @ homography
    lambda1 = 1 / np.linalg.norm(hprim[:, 0])
    lambda2 = 1 / np.linalg.norm(hprim[:, 1])
    lambda3 = np.sqrt(lambda1 * lambda2)

    r0 = hprim[:, 0] * lambda1
    r1 = hprim[:, 1] * lambda2
    r2 = np.cross(r0, r1)

    R = np.column_stack((r0, r1, r2))
    u, s, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, 2] *= -1
        R = u @ vh
    rotation = R
    t = hprim[:, 2] * lambda3
    position = -np.transpose(R) @ t
    return calibration, rotation, position


def estimate_calibration_matrix_from_plane_homography(homography, u, v):
    """
    This method initializes the calibration matrix from the homography between the world plane of the pitch
    and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
    Multiple View Geometry in computer vision, p225). The extraction is sensitive to noise, which is why we keep the
    principal point in the middle of the image rather than using the one extracted by this method.
    :param homography: homography between the world plane of the pitch and the image
    """
    H = np.reshape(homography, (9,))
    A = np.zeros((5, 6))
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    A[1, 2] = -1.0
    A[2, 3] = v / u
    A[2, 4] = -1.0
    A[3, 0] = H[0] * H[1]
    A[3, 1] = H[0] * H[4] + H[1] * H[3]
    A[3, 2] = H[3] * H[4]
    A[3, 3] = H[0] * H[7] + H[1] * H[6]
    A[3, 4] = H[3] * H[7] + H[4] * H[6]
    A[3, 5] = H[6] * H[7]
    A[4, 0] = H[0] * H[0] - H[1] * H[1]
    A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
    A[4, 2] = H[3] * H[3] - H[4] * H[4]
    A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
    A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
    A[4, 5] = H[6] * H[6] - H[7] * H[7]

    u, s, vh = np.linalg.svd(A)
    w = vh[-1]
    W = np.zeros((3, 3))
    W[0, 0] = w[0] / w[5]
    W[0, 1] = w[1] / w[5]
    W[0, 2] = w[3] / w[5]
    W[1, 0] = w[1] / w[5]
    W[1, 1] = w[2] / w[5]
    W[1, 2] = w[4] / w[5]
    W[2, 0] = w[3] / w[5]
    W[2, 1] = w[4] / w[5]
    W[2, 2] = w[5] / w[5]

    try:
        Ktinv = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        K = np.eye(3)
        return False, K

    K = np.linalg.inv(np.transpose(Ktinv))
    K /= K[2, 2]
    return True, K


def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = -first_tilt

    sign_first_tilt = 1.0 if np.sin(first_tilt) > 0.0 else -1.0
    sign_second_tilt = 1.0 if np.sin(second_tilt) > 0.0 else -1.0

    first_pan = np.arctan2(
        sign_first_tilt * orientation[0, 2], sign_first_tilt * -orientation[1, 2]
    )
    second_pan = np.arctan2(
        sign_second_tilt * orientation[0, 2], sign_second_tilt * -orientation[1, 2]
    )
    first_roll = np.arctan2(
        sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1]
    )
    second_roll = np.arctan2(
        sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1]
    )

    # print(f"first solution {first_pan*180./np.pi}, {first_tilt*180./np.pi}, {first_roll*180./np.pi}")
    # print(f"second solution {second_pan*180./np.pi}, {second_tilt*180./np.pi}, {second_roll*180./np.pi}")
    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


if __name__ == "__main__":
    template = utils.get_court_template()
    camera_params = np.array([640, 360, 2750, -10.5, -15.4, 0.0, 50.0, -39.0, 18.0])
    camera = Camera(camera_params)
    edge_map = camera.to_edge_map(template)
    utils.show_image([edge_map])
    params = decompose_homography_to_parameters(camera.homography(), 640, 360)
    print(
        f"camera params: [{params[0]}, {params[1]}, {params[2]:.2f}, {params[3]:.2f}, {params[4]:.2f}, {params[5]:.2f}, {params[6]:.2f}, {params[7]:.2f}, {params[8]:.2f},"
    )
    print("gr")
