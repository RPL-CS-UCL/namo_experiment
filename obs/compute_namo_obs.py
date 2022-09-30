import cv2
from utils.torch_utils import *
from obs.marker_detector import MarkerDetectorMulti
from obs.compute_occ_grid import OccGridComputer

import torch
import json
from datetime import datetime


def rotate_point(x, y, theta):
    out_x = x * np.cos(theta) - y * np.sin(theta)
    out_y = y * np.cos(theta) + x * np.sin(theta)

    return out_x, out_y


class NamoObsComputer():

    def __init__(self, markers, save=False, record=False, grid_size=64, goal_pos=[-1.5, 2]):
        '''
        Computes input vector required for the namo network by reading from two cameras
        '''
        # global variables
        self.grid_size = grid_size
        self.record = record
        self.dt = 0.016
        self.control_freq_inv = 20
        self.control_dt = self.dt * self.control_freq_inv
        self.device = 'cpu'
        self.goal_pos = to_torch(goal_pos)

        # init
        self.marker_detector_multi = MarkerDetectorMulti(
            markers, record=record)
        self.occ_grid_computer = OccGridComputer(
            grid_size=self.grid_size, goal_pos=self.goal_pos)
        self.detections = None
        self.occ_grid = torch.zeros(
            (1, 1, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)

        # obs states
        self.prev_robot_state = torch.zeros(
            3, dtype=torch.float32, device=self.device)
        self.curr_robot_state = torch.zeros(
            3, dtype=torch.float32, device=self.device)
        self.boxes_state = torch.zeros(
            (5, 3), dtype=torch.float32, device=self.device)
        self.box_status = torch.zeros(
            (1, 5), dtype=torch.bool, device=self.device)

        # normalization params
        self.max_room_dist = 8
        self.max_vel = 0.8
        self.max_rot_vel = 0.8

        num_obs = 48*5 + 2 + self.grid_size**2
        self.obs_buf = torch.zeros(
            (1, num_obs), dtype=torch.float32, device=self.device)

        self.save = save
        if self.save:
            timestamp = datetime.today().strftime('%m-%d-%H-%M')
            self.filename = f'saved_data/data_{timestamp}.npy'
            self.full_arr = np.ones((1, 6, 3))

    def get_input(self):
        '''
        Get detection of aruco marker poses from cameras
        '''
        self.detections = self.marker_detector_multi.detect()
        if self.save:
            with open(self.filename, 'wb') as f:
                self.full_arr = np.concatenate(
                    (self.full_arr, np.expand_dims(self.detections, 0)), axis=0)
                np.save(f, self.full_arr)

    def preprocess_input(self, actions):
        '''
        Pre-process the input into the desired vector format
        '''
        self.actions = actions

        self.occ_grid = self.occ_grid_computer.update(self.detections)
        self.occ_grid_computer.plot()

        # process detection
        self.curr_robot_state = to_torch(self.detections[5])
        self.boxes_state = to_torch(self.detections[0:5])

        pos = self.curr_robot_state[0:2]
        prev_pos = self.prev_robot_state[0:2]
        # rotate pos by 90
        # pos[0], pos[1] = rotate_point(pos[0], pos[1], np.pi/2)
        # prev_pos[0], prev_pos[1] = rotate_point(prev_pos[0], prev_pos[1], np.pi/2)

        rot = self.curr_robot_state[2].unsqueeze(0)
        prev_rot = self.prev_robot_state[2].unsqueeze(0)
        vel = torch.sub(pos, prev_pos)/self.control_dt
        ang_vel = torch.sub(rot, prev_rot)/self.control_dt
        actions = self.actions

        self.prev_robot_state = self.curr_robot_state.clone()

        # box keypoints
        self.box_status = torch.where(
            self.boxes_state[:, 0] == 99, torch.zeros_like(self.box_status), torch.ones_like(self.box_status))
        box_gym_state = process_box_state(self.boxes_state)
        boxes_keypoints_buf = torch.ones(
            1, 5, 8, dtype=torch.float32)

        temp_box_state = box_gym_state[:, 0:7].unsqueeze(0)
        # temp_box_state
        for i in range(5):
            box_keypoints = (torch.flatten(
                gen_keypoints(temp_box_state[:, i, :])[:, 0:4, 0:2], start_dim=1))
            boxes_keypoints_buf[:, i, :] = box_keypoints
        boxes_keypoints_buf *= self.box_status.unsqueeze(-1)
        boxes_keypoints = torch.flatten(
            boxes_keypoints_buf, start_dim=1)

        # normalize states
        pos /= (self.max_room_dist/2)
        vel /= self.max_vel
        rot /= np.pi
        ang_vel /= self.max_rot_vel

        boxes_keypoints /= (self.max_room_dist/2)

        curr_obs_buf = torch.cat(
            (pos.unsqueeze(0), vel.unsqueeze(0), rot.unsqueeze(0), ang_vel.unsqueeze(0), actions.unsqueeze(0), boxes_keypoints), dim=-1)

        # print(f'pos:     {pos}')
        # print(f'vel:     {vel}')
        # print(f'rot:     {rot}')
        # print(f'ang_vel: {ang_vel}')
        # print(f'boxes:   {self.boxes_state}')
        # print(f'keypts:  {boxes_keypoints}')

        num_obs = curr_obs_buf.shape[1]
        grid_size = self.grid_size
        num_channels = 1
        grid_buffer = grid_size * grid_size * num_channels
        num_history = 5

        if num_history > 1:
            for i in range(num_history-1):
                self.obs_buf[:, grid_buffer + i*num_obs:grid_buffer +
                             (i+1)*num_obs] = self.obs_buf[:, grid_buffer + (i+1) * num_obs:grid_buffer + (i+2)*num_obs]
            i = max(0, num_history - 2)
            self.obs_buf[:, grid_buffer + (i+1)*num_obs:grid_buffer +
                         (i+2)*num_obs] = curr_obs_buf[:]
        self.obs_buf[:, -
                     2:] = self.goal_pos.unsqueeze(0)/(self.max_room_dist/2)

        # print(f'goal:     {self.goal_pos/self.max_room_dist}')

        # print(self.obs_buf[:, grid_buffer:])

        rotated_grid = torch.rot90(self.occ_grid, 1, [2, 3])
        self.obs_buf[:, :grid_buffer] = torch.flatten(rotated_grid, 1, 3)

        return self.obs_buf

    def check_terminate(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.terminate()
            return

    def terminate(self):
        '''
        closes cv2 and camera connections
        '''
        print('terminates observer and saving file')
        self.marker_detector_multi.close()
        if self.record:
            self.cap.release()
            self.out_vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    namo_obs_computer = NamoObsComputer()
    while True:

        # time.sleep(1)
        action = torch.rand(2)
        namo_obs_computer.get_input(action)
        namo_obs_computer.preprocess_input()

        if namo_obs_computer.check_terminate():
            break

    cv2.destroyAllWindows()
