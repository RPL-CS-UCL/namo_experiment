import cv2
import sys
from utils.aruco_utils import ARUCO_DICT
import argparse
import time
import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.device_manager import DeviceManager
from obs.transform_frame import transform_frame
from datetime import datetime

from pykalman import KalmanFilter


class MarkerDetectorMulti():
    def __init__(self, markers, record=False):
        '''
        Handles two camera devices to detects marker poses in the scene w.r.t. a single referance marker
        '''
        ap = argparse.ArgumentParser()
        # change the matrices for if a different camera is used
        ap.add_argument("-k", "--K_Matrix", required=False,
                        help="Path to calibration matrix (numpy file)", default='utils/calibration_matrix_from_calib_py.npy')
        ap.add_argument("-d", "--D_Coeff", required=False,
                        help="Path to distortion coefficients (numpy file)", default='utils/distortion_coefficients_from_calib_py.npy')
        # change marker type of other marker types are used
        ap.add_argument("-t", "--type", type=str,
                        default="DICT_4X4_250", help="Type of ArUCo tag to detect")
        self.args = vars(ap.parse_args())

        if ARUCO_DICT.get(self.args["type"], None) is None:
            print(f"ArUCo tag type '{self.args['type']}' is not supported")
            sys.exit(0)

        c = rs.config()
        c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.device_manager = DeviceManager(rs.context(), c)
        self.device_manager.enable_all_devices()

        # define devices
        self.device1_id = '046122251230'
        self.device2_id = '138322252703'

        self.marker_detector1 = MarkerDetector(markers, 1, record, self.args)
        self.marker_detector2 = MarkerDetector(markers, 2, record, self.args)

        num_markers = len(markers.keys()) - 1

        self.markers_obs = np.ones((3, num_markers, 3), dtype=np.float)
        self.markers_obs[:, :, 0] = 99
        self.markers_obs[:, :, 1] = 99
        # self.stored_obs = self.markers_obs.copy()
        # self.markers_obs = np.ma.array(self.markers_obs)

        self.id_map = []  # maps id to array index of markers_obs
        self.kfs = []
        self.transition_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.observation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for key in markers.keys():
            if markers[key]['isReferenceMarker']:
                # don't append the reference marker
                continue
            self.id_map.append(key)
            # kf = KalmanFilter(transition_matrices=self.transition_matrix,
            #                   observation_matrices=self.observation_matrix)
            # self.kfs.append(kf)

    def detect(self):
        frames = self.device_manager.poll_frames()
        frame1 = list(frames[(self.device1_id, 'D400')].values())[0]
        frame2 = list(frames[(self.device2_id, 'D400')].values())[0]
        _, detection1, _ = self.marker_detector1.detect(frame1)
        _, detection2, _ = self.marker_detector2.detect(frame2)

        for (i, id) in enumerate(self.id_map):
            if detection1[id]['detected'] and detection1[id]['Pf'] != [] and detection1[id]['Rf'] != []:
                x = detection1[id]['Pf'][0]  # x
                y = detection1[id]['Pf'][1]  # y
                r = R.from_matrix(detection1[id]['Rf']).as_euler('zyx')[0]
                self.markers_obs[0][i][0] = x
                self.markers_obs[0][i][1] = y
                self.markers_obs[0][i][2] = r
            else:
                self.markers_obs[0][i] = [-99, -99, 0]
        for (i, id) in enumerate(self.id_map):
            if detection2[id]['detected'] and detection2[id]['Pf'] != [] and detection2[id]['Rf'] != []:
                x = detection2[id]['Pf'][0]  # x
                y = detection2[id]['Pf'][1]  # y
                r = R.from_matrix(detection2[id]['Rf']).as_euler('zyx')[0]
                self.markers_obs[1][i][0] = x
                self.markers_obs[1][i][1] = y
                self.markers_obs[1][i][2] = r
            else:
                self.markers_obs[1][i] = [-99, -99, 0]
        self.apply_average_filter()
        final_obs = self.markers_obs[2]
        return final_obs

    def apply_average_filter(self):
        # loop through objects
        for i in range(self.markers_obs.shape[1]):
            detection1 = self.markers_obs[0, i]
            detection2 = self.markers_obs[1, i]
            d1_is_masked = detection1[0] == -99
            d2_is_masked = detection2[0] == -99
            if d1_is_masked and d2_is_masked:
                pass
            elif d1_is_masked and not d2_is_masked:
                self.markers_obs[2][i] = detection2
            elif d2_is_masked and not d1_is_masked:
                self.markers_obs[2][i] = detection1
            else:
                # print(self.markers_obs[0:2, i])
                # will ignore masks
                self.markers_obs[2, i] = np.mean(
                    self.markers_obs[0:2, i], axis=0)
                # print(self.markers_obs[2][i])
        return

    def apply_kalman_filter(self):

        return

        for i in range(1):
            measurements = self.markers_obs[:, i, :]
            self.kfs[i] = self.kfs[i].em(measurements, n_iter=5)
            (filtered_state_means, filtered_state_covariances) = self.kfs[i].filter(
                measurements)
            (smoothed_state_means, smoothed_state_covariances) = self.kfs[i].smooth(
                measurements)
            print(measurements)
            print(filtered_state_means)
            # print(smoothed_state_means)

    def close(self):
        self.device_manager.disable_streams()


class MarkerDetector():
    def __init__(self, marker_description, camera_id, record, args):
        ''' 
        Marker detector class to detect arbitrary number of pre-defined markers. 
        
        Args:
            marker_description: Object containing marker information (see markers.json)
            camera_id: Device id
            record: set true to save camera feed into a video
        '''

        calibration_matrix_path = args["K_Matrix"]
        distortion_coefficients_path = args["D_Coeff"]

        self.k = np.load(calibration_matrix_path)
        self.d = np.load(distortion_coefficients_path)
        self.aruco_dict_type = ARUCO_DICT[args["type"]]

        self.marker_description = marker_description
        self.camera_id = camera_id

        self.record = record
        if self.record:
            timestamp = datetime.today().strftime('%m-%d-%H-%M')
            file_name = f'saved_videos/camera{camera_id}_{timestamp}.avi'
            self.out_vid = cv2.VideoWriter(
                file_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))

    def detect(self, frame, cap=None):
        '''
        Takes a camera frame and detects markers that are in the scene
        '''
        self.frame = frame
        self.color_image = np.asanyarray(self.frame.get_data())
        ret, detections, self.out_frame = transform_frame(
            self.color_image, self.aruco_dict_type, self.k, self.d, self.marker_description, self.camera_id)
        frame_y_pos = int(self.camera_id-1) * 500
        cv2.imshow(f'Estimated Pose{self.camera_id}', self.out_frame)
        cv2.moveWindow(f'Estimated Pose{self.camera_id}', 200, frame_y_pos)
        if self.record:
            self.out_vid.write(self.out_frame)
        return ret, detections, self.out_frame


if __name__ == '__main__':
    # input marker
    markers = {
        0: {
            'name': 'reference_marker',
            'marker_dim': 0.175,
            'isReferenceMarker': True
        },
        1: {
            'name': 'box1',
            'marker_dim': 0.175,
            'isReferenceMarker': False
        },
        2: {
            'name': 'box2',
            'marker_dim': 0.175,
            'isReferenceMarker': False
        },
        3: {
            'name': 'box3',
            'marker_dim': 0.175,
            'isReferenceMarker': False
        },
        4: {
            'name': 'box4',
            'marker_dim': 0.175,
            'isReferenceMarker': False
        }
    }

    marker_detector_multi = MarkerDetectorMulti(markers)
    while True:
        # time.sleep(1)
        marker_detector_multi.detect()
        if marker_detector_multi.close():
            break

    cv2.destroyAllWindows()
    marker_detector_multi.close()
