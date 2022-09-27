'''
Sample Usage:-
python transform_frame.py

This script is refactored from transform_to_franka_frame.py
'''


import numpy as np
import cv2
import sys
from utils.aruco_utils import ARUCO_DICT
import argparse
import time
import pyrealsense2 as rs
from numpy import *
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from utils.device_manager import DeviceManager

from pykalman import KalmanFilter

# example markers description
example_markers = {
    0: {  # id that the marker is generated from
        'name': 'best_marker',
        'marker_dim': 0.1,  # actual size of marker in mm
        'isReferenceMarker': True,  # only one marker can be the reference!
    }
}

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


def transform_frame(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, markers, camera_idx):
    '''
    frame - Frame from the video stream
    aruco_dict_type - Specifies which aruco type (e.g. 5x5)
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    markers - Dictionary containing some data about each marker (see example above)

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Correct distortion
    h1, w1 = gray.shape[:2]
    

    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict, parameters=parameters)

    def ret(): return 0

    # define some global variables
    detected_reference = False
    reference_Pc = []
    reference_Rc_inv = []

    detections = dict.fromkeys(markers.keys())
    for id, value in detections.items():
        detections[id] = {
            'Pf': [],
            'Rf': [],
            'Pc': [],
            'Rc': [],
            'detected': False
        }

    # If markers are detected
    ids = np.squeeze(ids, axis=-1)
    marker_size = [0.17, 0.1, 0.1]

    if len(corners) > 0:
        for (i, id) in enumerate(ids):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            try:
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], markers[id]['marker_dim'], matrix_coefficients,
                                                                               distortion_coefficients)
            except:
                print('skipping a step')
                continue

            # Draw a square around marker
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw axis on marker
            cv2.drawFrameAxes(frame, matrix_coefficients,
                              distortion_coefficients, rvec, tvec, 0.05)

            obj_Pc = np.squeeze(tvec)
            obj_Rc = np.squeeze(rvec)
            detections[id]['Rc'] = cv2.Rodrigues(obj_Rc)[0]
            detections[id]['Pc'] = obj_Pc
            detections[id]['detected'] = True
            is_reference = markers[id]['isReferenceMarker']

            if is_reference and not detected_reference:
                detected_reference = True
                reference_Pc = obj_Pc
                reference_Rc = obj_Rc
                reference_Rc_inv = np.linalg.inv(
                    cv2.Rodrigues(reference_Rc)[0])
            elif is_reference and detected_reference:
                print('Error: found two reference markers')

    else:
        pass
        # print("no marker detected")

    # cv2.imshow('Estimated Pose', frame)

    # adding correction since the franka marker is not centered in the robot frame origin but is translated
    # of 0.25 m along the x axis. actually using it as a calibration for fixing the translation error
    reference_x_correction = 0.0

    if not detected_reference:
        print('reference mark not detected')
    for (i, detection) in detections.items():
        if not detected_reference or markers[i]['isReferenceMarker']:
            pass
        elif detections[i]['detected'] and detected_reference:
            detections[i]['Pf'] = reference_Rc_inv.dot(
                (detections[i]['Pc'] - reference_Pc))
            detections[i]['Pf'][0] = detections[i]['Pf'][0] + \
                reference_x_correction
            detections[i]['Rf'] = np.matmul(
                reference_Rc_inv, detections[i]['Rc'])
            marker_name = markers[i]['name']

            # print(
            #     f'{camera_idx}: Final position of {marker_name} in reference world=', detections[i]['Pf'])

    return ret, detections, frame


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=False,
                    help="Path to calibration matrix (numpy file)", default='calibration_matrix_from_calib_py.npy')
    ap.add_argument("-d", "--D_Coeff", required=False,
                    help="Path to distortion coefficients (numpy file)", default='distortion_coefficients_from_calib_py.npy')
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    # d = distortion_coeff

    c = rs.config()
    c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    device_manager = DeviceManager(rs.context(), c)
    device_manager.enable_all_devices()
    while True:
        frames = device_manager.poll_frames()
        # print(frames)
        # frame1 = list(frames[('046122251230', 'D400')].values())[0]
        frame2 = list(frames[('138322252703', 'D400')].values())[0]
        # color_frame = frame1.get_color_frame()
        # color_image1 = np.asanyarray(frame1.get_data())
        color_image2 = np.asanyarray(frame2.get_data())
        # print('camera1:')
        # ret, detections, out_frame = transform_frame(
        #     color_image1, aruco_dict_type, k, d, markers, 1)
        # print('camera2:')
        ret2, detections2, out_frame2 = transform_frame(
            color_image2, aruco_dict_type, k, d, markers, 2)

        # cv2.imshow('Estimated Pose', out_frame)
        cv2.imshow('Estimated Pose2', out_frame2)
        # print(detections)
        # print(detections[0]['Pf'])
        # print(detections[0]['Pc'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    quit()
