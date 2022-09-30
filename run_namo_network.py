

import sys
import utils.gamepad_reader as pad
import robot_interface as bot

from obs.compute_namo_obs import NamoObsComputer
from network.load_namo_network import post_process, load_namo_network

import torch
import json
import time
from threading import Thread, Lock
import cv2

# read marker information from json file
with open("obs/markers.json") as f:
    markers = json.load(f)

class NAMOController:
    def __init__(self, save=False, record=False):
        # relative_network_path = '../summit_xl_gym/runs/test_map2_noRot/nn/ep_300.pth'
        relative_network_path = '../summit_xl_gym/runs/test_map_noRot/nn/ep750.pth'
        
        goal_pos = [-1.5, 2]

        self.observer = NamoObsComputer(grid_size=48, goal_pos=goal_pos, save=save, record=record)
        self.net = load_namo_network(path=relative_network_path)

        # set network to evaluation mode
        self.net.eval()

        # run one forward pass to initiate everything
        with torch.no_grad():
            self.obs = torch.rand(1, 2546)
            # forward pass
            mu, logstd, _, _ = self.net(self.obs)
            # post process
            self.action = post_process(mu, logstd)
            print(self.action)

    def get_obs(self):
        '''
        gets detection from camera
        '''
        self.observer.get_input()

    def preprocess_input(self, action):
        '''
        format latest detection into observation
        '''
        self.obs = self.observer.preprocess_input(self.action)

    def get_cmd(self):
        '''
        returns (vx, wz)
        '''
        # return 0.5, 0

        # fills self.obs
        self.preprocess_input(self.action)

        # random_obs = torch.rand(1, 2546)
        with torch.no_grad():
            mu, logstd, _, _ = self.net(self.obs)
            # mu, logstd, _, _ = self.net(self.obs)
            action = post_process(mu, logstd)
            vx = action[1]
            zw = action[0]
        
        vx_mult = 0.65
        if vx < 0:
            vx_mult *= 0.3
            
        return -zw * 0.5, vx * vx_mult + 0.025

    def check_complete(self):
        '''
        reads relative position between robot and goal from self.obs
        '''
        pass


class padCmd:
    def __init__(self):
        self.vx = 0
        self.vy = 0
        self.wz = 0
        self.height = 0.95
        self.overwrite = False
        self.terminate = True


class LatestCmd:
    def __init__(self, lock):
        '''
        Sets control cmds from the network and gamepad
        '''
        self.latest_cmd = padCmd()

        self.lock = lock

    def set_cmd(self, vx, vy, wz):
        with self.lock:
            self.latest_cmd.vx = vx
            self.latest_cmd.vy = vy
            self.latest_cmd.wz = wz
        return

    def get_cmd(self):
        with self.lock:
            return self.latest_cmd


class padCallback:
    def __init__(self):

        real_max = 255
        real_min = 0
        sign = ['minus', 'minus', 'plus']
        correction = pad.Correction(real_max, real_min, sign)
        self.terminate = False
        try:
            self.gamepad = pad.Gamepad(dead_zone=0.004, vel_scale_x=1., vel_scale_y=1., vel_scale_rot=1.,
                                       correction=correction)
        except:
            sys.exit("no controller connected to the computer. attach one")

    def GetGamePadCmd(self):
        ret = padCmd()
        ret.vx = self.gamepad.vx
        ret.vy = self.gamepad.vy
        ret.wz = self.gamepad.wz
        ret.overwrite = self.gamepad.overwrite
        self.terminate = self.gamepad.estop_flagged
        return ret, self.terminate

# define global variables
commandLock = Lock()
globalgamepad = padCallback()
namo_controller = NAMOController(save=True, record=True)
latest_cmd = LatestCmd(commandLock)


def controlLogicGamepad(state, init_state, t):
    '''
    directly sends control signals to robot, runs in main thread
    '''

    gamepad_cmd, terminate = globalgamepad.GetGamePadCmd()
    overwrite = gamepad_cmd.overwrite
    # overwrite = True
    network_cmd = latest_cmd.get_cmd()

    if terminate or overwrite:
        control_cmd = gamepad_cmd
    else:
        control_cmd = network_cmd

    print(f'{control_cmd.vx}, {control_cmd.vy}, {control_cmd.wz}')
    # print(f'{gamepad_cmd.vx}, {gamepad_cmd.vy}, {gamepad_cmd.wz}')
    # print("t in milliseconds=", t)
    cmd = bot.HighCmdSimple()
    cmd.mode = 0
    cmd.gait_type = 0
    cmd.speed_level = 0
    cmd.foot_raise_height = 0
    cmd.body_height = gamepad_cmd.height  # -0.1 to make it lower
    myeuler = [0] * 3
    cmd.euler = myeuler
    myvelocity = [0] * 2
    cmd.velocity = myvelocity
    cmd.yaw_speed = 0.0

    if 0 < t < 40000:
        cmd.mode = 2
        cmd.gait_type = 2
        myvelocity[0] = control_cmd.vx
        myvelocity[1] = control_cmd.vy
        cmd.velocity = myvelocity  # -1  ~ +1
        cmd.yaw_speed = control_cmd.wz
        cmd.foot_raise_height = 0.1
    if t >= 40000:
        cmd.mode = 1
        cmd.running_controller = False
        print("step closing controller")
    if terminate:
        cmd.mode = 1
        cmd.running_controller = False

    return cmd


def run_control_thread():
    '''
    The secondary thread which receives commands from the network at 3Hz
    '''
    print('starting thread')
    control_freq_inv = 1/3
    t = 0
    while True:
        t0 = time.time()
        # get observation
        namo_controller.get_obs()
        print('secondary thread running')
        # print(f'time at start: {start}')

        if t > control_freq_inv:
            start = time.time()
            # preprocess observation & forward pass
            zw, vx, = namo_controller.get_cmd()
            # print(zw.item(), vx.item())

            # set commands
            latest_cmd.set_cmd(vx, 0, zw)
            end = time.time()
            t = end-start
        # to maintain control_freq_inv between each calls
        # print(f'time at done: {end}')
        # print(f'time taken:   {time_taken}')

        t1 = time.time()
        t += (t1-t0)

        if namo_controller.observer.check_terminate():
            break

    cv2.destroyAllWindows()


memory_example = True


if __name__ == "__main__":
    print("Communication level is set to HIGH-level.")
    print("WARNING: Make sure the robot is on the ground.")
    print("Press Enter to continue...")
    input()

    object_methods = [method_name for method_name in dir(bot)
                      if callable(getattr(bot, method_name))]

    thread = Thread(target=run_control_thread)
    thread.start()

    # ------------------test without the robot ------------------
    # start_t = time.time()
    # while True:
    #     t = time.time()
    #     controlLogicGamepad(None, None, t=t-start_t)

    # ------------------running on the robot-----------------
    robot = bot.HIGO1_()
    if (memory_example):
        # example with memory (using objects created inside the python script)
        robot.set_controller(controlLogicGamepad)
    else:
        pass

    # executing control callback
    robot.run()
    # stop gamepad thread
    globalgamepad.gamepad.stop()
