'''
This script contains the function load_namo_network to intialize and load the ]
custom multi-modal neural network from pretrained weights.

rl_games needs to be installed in the environment to run this script
'''

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import time

# parameters for generating the network
params = {
    'name': 'custom_net',
    'normalization': 'layer_norm',
    'separate': False,
    'space': {
        'continuous': {
            'mu_activation': 'None',
            'sigma_activation': 'None',
            'mu_init': {
                'name': 'default'},
            'sigma_init': {
                'name': 'const_initializer',
                'val': 0},
            'fixed_sigma': True}},
    'mlp': {
        'units': [128],
        'activation': 'relu',
        'd2rl': False,
        'initializer': {'name': 'default'},
        'regularizer': {'name': 'l2_regularizer'},
        'dropout': {'p': 0}},
    'cnn': {
        'type': 'conv2d',
        'grid_size': 48,
        'num_input_channels': 1,
        'initializer': {'name': 'default'},
        'regularizer': {'name': 'l2_regularizer'},
        'activation': 'relu',
        'convs': [{
                  'filters': 16,
                  'kernel_size': 8,
                  'strides': 4,
                  'padding': 0,
                  'dropout': {
                      'p': 0}},
                  {
                  'filters': 32,
                  'kernel_size': 4,
                  'strides': 2,
                  'padding': 0,
                  'dropout': {'p': 0}
                  }],
        'intermediate_fcl': {
            'units': [128],
            'activation': 'relu',
            'dropout': {'p': 0}}},
    'fusion_mlp': {
        'units': [128, 64],
        'activation': 'relu',
        'd2rl': False,
        'initializer': {'name': 'default'},
        'regularizer': {'name': 'l2_regularizer'},
        'dropout': {'p': 0}
    }}

# some other parameters that the network builder expects
other_params = {'actions_num': 2, 'input_shape': (
    2546,), 'num_seqs': 2048, 'value_size': 1}


def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(
                f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
            wait_sec = 2 ** attempt
            print(f'Waiting {wait_sec} before trying again...')
            time.sleep(wait_sec)

    raise RuntimeError(
        f'Could not execute {func}, give up after {num_attempts} attempts...')


def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_filesystem_op(
        torch.load, filename, map_location=lambda storage, loc: storage.cuda(1))
    return state


def neglogp_func(x, mean, std, logstd):
    return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
        + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
        + logstd.sum(dim=-1)



class customNetBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        # print(network_builder)
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_intermediate_fcl = nn.Sequential()
            self.critic_intermediate_fcl = nn.Sequential()
            self.actor_fusion_mlp = nn.Sequential()
            self.critic_fusion_mlp = nn.Sequential()

            # DEFINE CNN
            self.grid_size = self.cnn['grid_size']
            self.cnn_num_input_channels = self.cnn['num_input_channels']
            cnn_input_shape = (self.cnn_num_input_channels,
                               self.grid_size, self.grid_size)
            cnn_args = {
                'ctype': self.cnn['type'],
                'input_shape': cnn_input_shape,
                'convs': self.cnn['convs'],
                'activation': self.cnn['activation'],
                'norm_func_name': "batch_norm",
            }
            self.actor_cnn = self._build_conv(**cnn_args)
            if self.separate:
                self.critic_cnn = self._build_mlp(**mlp_args)

            # DEFINE MLP
            mlp_input_shape = (
                input_shape[0]-self.grid_size**2*self.cnn_num_input_channels)
            mlp_out_size = self.mlp_units[-1]
            mlp_args = {
                'input_size': mlp_input_shape,
                'units': self.mlp_units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer,
                'dropout': self.mlp_dropout,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            # DEFINE INTERMEDIATE_FCL:
            if self.intermediate_fcl:
                units = self.intermediate_fcl['units']
                intermediate_fcl_input_shape = self._calc_input_size(
                    cnn_input_shape, self.actor_cnn)
                intermediate_fcl_out_size = units[-1]
                mlp_args = {
                    'input_size': intermediate_fcl_input_shape,
                    'units': units,
                    'activation': self.activation,
                    'norm_func_name': self.normalization,
                    'dense_func': torch.nn.Linear,
                    'd2rl': self.is_d2rl,
                    'norm_only_first_layer': self.norm_only_first_layer,
                    'dropout': self.intermediate_fcl['dropout']
                }
                self.actor_intermediate_fcl = self._build_mlp(
                    **mlp_args)
                if self.separate:
                    self.critic_intermediate_fcl = self._build_mlp(**mlp_args)

            # DEFINE FUSION MLP
            if self.intermediate_fcl:
                fusion_mlp_input_shape = intermediate_fcl_out_size
            else:
                fusion_mlp_input_shape = self._calc_input_size(
                    cnn_input_shape, self.actor_cnn)
            fusion_mlp_input_shape += mlp_out_size
            print('fusion input shape:')
            print(fusion_mlp_input_shape)
            out_size = self.fusion_mlp_units[-1]
            fusion_mlp_args = {
                'input_size': fusion_mlp_input_shape,
                'units': self.fusion_mlp_units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer,
                'dropout': self.fusion_mlp_dropout
            }
            self.actor_fusion_mlp = self._build_mlp(**fusion_mlp_args)
            if self.separate:
                self.critic_fusion_mlp = self._build_mlp(**fusion_mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(
                self.value_activation)
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList(
                    [torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(
                    self.space_config['mu_activation'])
                mu_init = self.init_factory.create(
                    **self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(
                    self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(
                    **self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(
                        actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            cnn_init = self.init_factory.create(**self.cnn['initializer'])

            # print('MODULES:')
            for m in self.modules():
                # print(m)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
            return

        def forward(self, obs):
            states = None
            cnn_input = obs[:, :self.grid_size **
                            2 * self.cnn_num_input_channels].reshape(-1, self.cnn_num_input_channels, self.grid_size, self.grid_size)


            mlp_input = obs[:, self.grid_size **
                            2 * self.cnn_num_input_channels:]
            if self.separate:
                print('not supported!')
                pass
            else:
                out_mlp = mlp_input
                out_mlp = self.actor_mlp(out_mlp)

                out_cnn = cnn_input
                out_cnn = self.actor_cnn(out_cnn)
                out_cnn = out_cnn.flatten(start_dim=1, end_dim=3)

                if self.intermediate_fcl:
                    out_cnn = self.actor_intermediate_fcl(out_cnn)

                out = torch.cat((out_mlp, out_cnn), dim=-1)
                out = self.actor_fusion_mlp(out)

                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states
                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def load(self, params):
            self.separate = params.get('separate', False)
            self.mlp_units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.mlp_dropout = params['mlp']['dropout']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get(
                'norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get(
                'joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.intermediate_fcl = self.cnn.get(
                    'intermediate_fcl', None)
            else:
                self.has_cnn = False

            if 'fusion_mlp' in params:
                self.has_fusion_mlp = True
                self.fusion_mlp_units = params['fusion_mlp']['units']
                self.fusion_mlp_dropout = params['fusion_mlp']['dropout']
            else:
                self.has_fusion_mlp = False

            return

    def build(self, name, **kwargs):
        net = customNetBuilder.Network(self.params, **kwargs)
        return net


def load_namo_network(path):
    
    # Loading trained model file
    trained_model_path = path
    all_parameters = load_checkpoint(trained_model_path)
    trained_model_parameters = all_parameters['model']

    # remove prefix from the parameter keys
    trained_model_parameters_renamed = {}
    for key in trained_model_parameters.keys():
        new_key = key.split('a2c_network.')[1]
        trained_model_parameters_renamed[new_key] = trained_model_parameters[key]

    # build the neural network
    model = customNetBuilder()
    model.load(params)
    net = model.build('custom_a2c', **other_params)
    
    # load network weights
    net.load_state_dict(trained_model_parameters_renamed)

    return net


def post_process(mu, logstd):
    # for non-determinstic policy outputs:
    # sigma = torch.exp(logstd)
    # distr = torch.distributions.Normal(mu, sigma)
    # selected_action = distr.sample()
    
    selected_action = mu

    # clamp action to -1 and 1
    action_tensor = torch.clamp(
        selected_action, -1, 1)

    return action_tensor.squeeze()


if __name__ == "__main__":
    
    # Initialise the network with weights
    # path = '../summit_xl_gym/runs/48G_48HL_8Rot/nn/ep100.pth'
    path = 'weights/test_map_ep750.pth'
    net = load_namo_network(path)

    # forward pass on random input
    example_input = torch.rand(1, 2546)
    mu, logstd, _, _ = net(example_input)

    # post process
    action = post_process(mu, logstd)

    print('---------------------------')
    print('ACTION OUTPUT:')
    print(action)
