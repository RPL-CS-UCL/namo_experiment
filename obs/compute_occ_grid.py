import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt



class OccGridComputer():
    def __init__(self, goal_pos=[-1, 1], grid_size=64):
        '''
        Class for computing the occupancy grid image given the preprocessed inputs.

        Note that the information about the walls are predefined and will not be changed. 
        The dimentions of boxes and robot are predefined, their pose data
        will be updated from the camera readings.
        '''
        self.grid_size = grid_size
        WALL_WIDTH = 0.2
        BOX_WIDTH = 0.6
        self.room_size = 8  # meters

        # allocate tensors
        self.occ_grid = torch.zeros(
            (1, 1, self.grid_size, self.grid_size), dtype=torch.float32)
        self.Px = torch.tensor(list(range(self.grid_size))).unsqueeze(
            dim=-1).unsqueeze(dim=-1).repeat(1, 1, 1, 1)
        self.Py = torch.transpose(self.Px, 1, 2)

        # state definition: [x_pos, y_pos, width, height, rot_in_radians]

        # robot
        self.robot_vertices = torch.ones((1, 1, 4, 2), dtype=torch.float)
        self.rotated_robot_vertices = torch.ones_like(self.robot_vertices)
        self.robot_occ_cells = torch.ones(
            (1, self.grid_size, self.grid_size), dtype=torch.float)

        self.robot_state = torch.tensor([[[-99, -99, 0.7, 0.63, 1e-8]]])

        # walls
        self.walls_vertices = torch.ones((1, 5, 4, 2), dtype=torch.float)
        self.rotated_walls_vertices = torch.ones_like(self.walls_vertices)
        self.walls_occ_cells = torch.ones(
            (1, self.grid_size, self.grid_size), dtype=torch.float)

        # test map 2 walls
        # self.walls_state = torch.tensor([[[0.5, 0.5, 3, WALL_WIDTH, np.pi/2+1e-8],
        #                                   [0, 3, 4.2, WALL_WIDTH, np.pi/2+1e-8],
        #                                   [-0.5, -3, 5.2, WALL_WIDTH, np.pi/2+1e-8],
        #                                   [-2, 1, 4.2, WALL_WIDTH, 1e-8],
        #                                   [2, 0, 6.2, WALL_WIDTH, 1e-8],
        #                                   [-3, -2, 2.2, WALL_WIDTH, 1e-8],
        #                                   [-2.5, -1, 1.2, WALL_WIDTH, np.pi/2+1e-8]]])
        
        # test map 1 walls
        self.walls_state = torch.tensor([[[0.7, 0, 2.6, WALL_WIDTH, np.pi/2+1e-8],
                                          [0, 2.5, 4.2, WALL_WIDTH, np.pi/2+1e-8],
                                          [0, -2.5, 4.2, WALL_WIDTH, np.pi/2+1e-8],
                                          [2, 0, 5.2, WALL_WIDTH, 1e-8],
                                          [-2, 0, 5.2, WALL_WIDTH, 1e-8]]])

        # boxes
        self.boxes_vertices = torch.ones((1, 5, 4, 2), dtype=torch.float)
        self.rotated_boxes_vertices = torch.ones_like(self.boxes_vertices)
        self.boxes_occ_cells = torch.ones(
            (1, self.grid_size, self.grid_size), dtype=torch.float)

        self.boxes_state = torch.tensor([[[-99, -99, BOX_WIDTH, BOX_WIDTH, 1e-8],
                                          [-99, -99, BOX_WIDTH, BOX_WIDTH, 1e-8],
                                          [-99, -99, BOX_WIDTH, BOX_WIDTH, 1e-8],
                                          [-99, -99, BOX_WIDTH, BOX_WIDTH, 1e-8],
                                          [-99, -99, BOX_WIDTH, BOX_WIDTH, 1e-8]]])

        # goal
        self.goal_vertices = torch.ones((1, 1, 4, 2), dtype=torch.float)
        self.rotated_goal_vertices = torch.ones_like(self.goal_vertices)
        self.goal_occ_cells = torch.ones(
            (1, self.grid_size, self.grid_size), dtype=torch.float)

        self.goal_state = torch.tensor(
            [[[goal_pos[0], goal_pos[1], 0.5, 0.5, 1e-8]]])

        # perform one update
        self.walls_state[:, :, 0] += self.room_size/2
        self.walls_state[:, :, 1] += self.room_size/2

        self.walls_vertices, self.rotated_walls_vertices = get_vertices(
            self.walls_state, self.walls_vertices, self.rotated_walls_vertices, n=self.grid_size)
        self.walls_occ_cells = get_occ_cells(
            self.Px, self.Py, self.walls_vertices, self.rotated_walls_vertices)

        self.goal_state[:, :, 0:2] += self.room_size/2
        self.goal_vertices, self.rotated_goal_vertices = get_vertices(
            self.goal_state, self.goal_vertices, self.rotated_goal_vertices, n=self.grid_size)
        self.goal_occ_cells = get_occ_cells(
            self.Px, self.Py, self.goal_vertices, self.rotated_goal_vertices)

    def update(self, detections):
        '''
        Updates the occupancy grid from processed camera readings (detections).
        '''
        for (i, detection) in enumerate(detections):
            if i < len(detections)-1:
                # print(detection)
                # box
                self.boxes_state[0][i][0] = detection[0] + self.room_size/2
                self.boxes_state[0][i][1] = detection[1] + self.room_size/2
                self.boxes_state[0][i][4] = detection[2]
            if i == len(detections) - 1:
                # robot
                self.robot_state[0][0][0] = detection[0] + self.room_size/2
                self.robot_state[0][0][1] = detection[1] + self.room_size/2
                self.robot_state[0][0][4] = detection[2]

        self.robot_vertices, self.rotated_robot_vertices = get_vertices(
            self.robot_state, self.robot_vertices, self.rotated_robot_vertices, n=self.grid_size)
        self.robot_occ_cells = get_occ_cells(
            self.Px, self.Py, self.robot_vertices, self.rotated_robot_vertices)

        self.boxes_vertices, self.rotated_boxes_vertices = get_vertices(
            self.boxes_state, self.boxes_vertices, self.rotated_boxes_vertices, n=self.grid_size)
        self.boxes_occ_cells = get_occ_cells(
            self.Px, self.Py, self.boxes_vertices, self.rotated_boxes_vertices)

        self.occ_grid = compute_occ_grid_mask(
            self.occ_grid, self.robot_occ_cells, self.boxes_occ_cells, self.walls_occ_cells, self.goal_occ_cells)

        return self.occ_grid

    def plot(self):
        # print("---- saving image ----")
        # plt.figure(figsize=(9, 4))
        # plt.tight_layout()
        # for i in range(1):
        #     ax = plt.subplot(1, 3, i + 1)
        #     ax.imshow(grid[0][i])
        #     ax.set_title(f'frame{i}')
        #     ax.grid(False)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.savefig('test_occ_grid.png')
        # grid = grid/3 * 255
        # print(grid)
        cv2.namedWindow("Occupancy grid", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Occupancy grid", 1000, -500)
        cv2.imshow(f'Occupancy grid', self.occ_grid[0][0].numpy())
        # plt.imshow(self.occ_grid[0][0].numpy())
        # plt.savefig('test_occ_grid.png')
        # plt.close()


def get_euler_z(q):
    qx, qy, qz, qw = 0, 1, 2, 3

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return yaw


def get_obj_states(obj_root_state, obj_name, obj_descriptions, offset, out):
    '''
    Extracts relevent information from obj_root_state

    Args
    -----------------------
    obj_root_state:   (num_envs x num_actors x 13)
    obj_name:          either 'walls' 'boxes' or 'summit'
    obj_descriptions:  Tensor containing object width & height

    Returns
    -----------------------
    out:              shape (num_envs x num_objects x 5)
    '''

    x_offset, y_offset = 0, 0

    out[:, :, 0] = obj_root_state[:, :, 0] + 5 + x_offset  # pos x
    out[:, :, 1] = obj_root_state[:, :, 1] + 5 + y_offset  # pos y
    out[:, :, 2] = obj_descriptions[:, :, 0]  # width
    out[:, :, 3] = obj_descriptions[:, :, 1]  # height

    # do a for loop here since get_euler_xyz only supports one object
    i = 0
    while i < obj_root_state.shape[1]:
        out[:, i, 4] = get_euler_z(obj_root_state[:, i, 3:7])
        i += 1
    return out


def get_vertices(obj_state, obj_vertices, obj_rot_vertices, room_size=8, n=64):
    '''
    Takes in information about rectangular objects in the environments and returns the vertices

    Args
    -----------------------
    obj_state:        object description in the environments, with shape (num_envs x num_objects x 5)
    obj_vertices:     tensor buffer for the vertices with shape (num_envs x num_objects x 4 x 2)
    obj_rot_vertices: tensor buffer for the rotated vertices with shape (num_envs x num_objects x 4 x 2)
    room_size:        room size in meters
    n:                grid size

    Returns
    -----------------------
    obj_vertices:     shape (num_envs x num_objects x 4 x 2)
    obj_rot_vertices: shape (num_envs x num_objects x 4 x 2)
    '''
    w = obj_state[:, :, 2] * (n/room_size)
    h = obj_state[:, :, 3] * (n/room_size)
    d = ((w/2)**2 + (h/2)**2)**0.5
    num_obj = obj_state.shape[1]

    # only these need to be updated - object shape doesn't change
    x_pos = obj_state[:, :, 0] * (n/room_size)
    y_pos = obj_state[:, :, 1] * (n/room_size)
    theta = -(obj_state[:, :, 4] + np.pi/2)
    obj_vertices[:, :, 0, 0] = torch.sin(torch.arctan(w/h) + theta) * d + x_pos
    obj_vertices[:, :, 0, 1] = torch.cos(torch.arctan(w/h) + theta) * d + y_pos
    obj_vertices[:, :, 1, 0] = torch.sin(
        np.pi - torch.arctan(w/h) + theta) * d + x_pos
    obj_vertices[:, :, 1, 1] = torch.cos(
        np.pi - torch.arctan(w/h) + theta) * d + y_pos
    obj_vertices[:, :, 2, 0] = torch.sin(
        np.pi + torch.arctan(w/h) + theta) * d + x_pos
    obj_vertices[:, :, 2, 1] = torch.cos(
        np.pi + torch.arctan(w/h) + theta) * d + y_pos
    obj_vertices[:, :, 3,
                 0] = torch.sin(-torch.arctan(w/h) + theta) * d + x_pos
    obj_vertices[:, :, 3,
                 1] = torch.cos(-torch.arctan(w/h) + theta) * d + y_pos

    obj_rot_vertices[:, :, 1, :] = obj_vertices[:, :, 0, :]
    obj_rot_vertices[:, :, 2, :] = obj_vertices[:, :, 1, :]
    obj_rot_vertices[:, :, 3, :] = obj_vertices[:, :, 2, :]
    obj_rot_vertices[:, :, 0, :] = obj_vertices[:, :, 3, :]

    return obj_vertices, obj_rot_vertices


def get_occ_cells(Px, Py, vertices, rotated_vertices):
    '''
    Computes a binary occupancy cell from object vertices
    '''
    num_obj = vertices.shape[1]
    vertices = vertices.view(-1, 1, 1, num_obj*4, 2)
    rotated_vertices = rotated_vertices.view(-1, 1, 1, num_obj*4, 2)
    n = Py.shape[1]
    num_envs = Py.shape[0]
    b = vertices - rotated_vertices
    device = Px.device

    vertex_mask = torch.where(torch.mul(torch.sub(Px, vertices[..., 0]), b[..., 1]) -
                              torch.mul(torch.sub(Py, vertices[..., 1]), b[..., 0]) <= 0, 0, 1)

    mask_buf = torch.zeros(
        (num_envs, n, n), device=device, dtype=torch.float32)
    i = 0
    while i < vertex_mask.shape[-1]:
        temp_mask = torch.ones_like(mask_buf)
        for j in range(4):
            temp_mask = torch.mul(temp_mask, vertex_mask[..., i+j])
        # overwrite to mask
        mask_buf = torch.where(
            temp_mask != 0, torch.ones_like(mask_buf), mask_buf)
        i += 4

    return mask_buf


def compute_occ_grid_mask(occ_grid, summit_occ_cells, boxes_occ_cells, walls_occ_cells, goal_occ_cells):
    '''
    computes the final occupancy grid with the following semantic data 

    0: free space
    1: walls
    2: boxes
    3: summit
    4: goal
    '''
    c = occ_grid.shape[1]
    for i in range(occ_grid.shape[1]-1):
        occ_grid[:, i, :, :] = occ_grid[:, i+1, :, :]

    # 0: free space
    # 1: walls
    # 2: boxes
    # 3: summit
    # 4: goal

    # priority: wall > summit > boxes > goal
    occ_grid[:, c-1, :, :] = boxes_occ_cells * 2
    occ_grid[:, c-1, :, :] = torch.where(goal_occ_cells > 0,
                                         goal_occ_cells * 4, occ_grid[:, c-1, :, :])
    occ_grid[:, c-1, :, :] = torch.where(summit_occ_cells > 0,
                                         summit_occ_cells * 3, occ_grid[:, c-1, :, :])
    occ_grid[:, c-1, :, :] = torch.where(walls_occ_cells > 0,
                                         walls_occ_cells, occ_grid[:, c-1, :, :])

    # normalize
    occ_grid /= 4

    return occ_grid
