from obs.compute_namo_obs import NamoObsComputer
from network.load_namo_network import post_process, load_namo_network

import torch
import json
import cv2


if __name__ == '__main__':
    
    # Markers definitions
    # Note that only one marker must be the reference marker, and make sure it can be captured by all cameras
    # Marker dim is defined in meters for the width of the square markers.

    # read marker information from json file
    with open("obs/markers.json") as f:
        markers = json.load(f)
    
    namo_obs_computer = NamoObsComputer(markers, grid_size=48, goal_pos=[-1.5, 2], save=False, record=False)
    net = load_namo_network(path='../summit_xl_gym/runs/test_map_noRot/nn/ep750.pth')

    net.eval()

    with torch.no_grad():
        print(net)
        for name, _ in net.named_parameters():
            print(name)

        example_input = torch.rand(1, 2546)
        # sample_obs = torch.load(
        #     '../summit_xl_gym/obs2.pt').unsqueeze(0).clone().to('cpu')
        # print(sample_obs)

        # forward pass
        mu, logstd, _, _ = net(example_input)
        print(mu)

        # post process
        action = post_process(mu, logstd)
        print(action)

    while True:

        # time.sleep(1/3)
        namo_obs_computer.get_input()
        obs = namo_obs_computer.preprocess_input(action)

        # print(obs.shape)
        with torch.no_grad():
            # forward pass
            mu, logstd, _, _ = net(obs)

            #
            # print(f'unprocessed action: {mu}')
            # print(logstd)

            # post process
            action = post_process(mu, logstd)
            print(f'action: {action}')

        if namo_obs_computer.check_terminate():
            break

    cv2.destroyAllWindows()
