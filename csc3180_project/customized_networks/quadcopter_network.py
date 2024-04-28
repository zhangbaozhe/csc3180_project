import torch
from torch import nn
import torch.nn.functional as F

class QuadcopterNetwork(nn.Module):
    """
    obs = physical (13D) + depth image (160x120x1)
    physical -> MLP
    depth image -> CNN
    """
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        print("QuadcopterNetwork")
        actions_num = kwargs.pop('actions_num', 4)
        input_shape = kwargs.pop('input_shape', (13+160*120,))

        self.central_value = params.get('central_value', False)
        self.value_size = params.get('value_size', 1)

        self.physical_mlp = nn.Sequential(
            nn.Linear(13, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 1, 4, 1), 
            nn.ReLU(),
            nn.Conv2d(1, 1, 32, 2), 
            nn.ReLU(), 
            nn.Flatten()
        )
        self.depth_fc = nn.Sequential(
            nn.Linear(2709, 256),
            nn.ReLU(), 
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(64+64, actions_num)
        self.value_linear = nn.Linear(64+64, self.value_size)

    def is_rnn(self):
        return False

    def forward(self, obs):
        batch_size = obs['obs'].shape[0]
        physical_obs = obs['obs'][..., 0:13]
        # from 1d to (1, 1, 120, 160)
        depth_image = obs['obs'][..., 13:].reshape(batch_size, 1, 160, 120)

        physical_out = self.physical_mlp(physical_obs)
        depth_image_out = self.depth_cnn(depth_image)
        depth_image_out = self.depth_fc(depth_image_out)

        mean_output = self.mean_linear(torch.cat([physical_out, depth_image_out], dim=1))
        value_output = self.value_linear(torch.cat([physical_out, depth_image_out], dim=1))
        if self.central_value:
            return value_output, None
        return mean_output, torch.zeros_like(mean_output), value_output, None


from rl_games.algos_torch.network_builder import NetworkBuilder

class QuadcopterNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
    
    def build(self, name, **kwargs):
        return QuadcopterNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)