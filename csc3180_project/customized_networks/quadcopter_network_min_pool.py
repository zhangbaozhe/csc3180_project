import torch
from torch import nn
import torch.nn.functional as F

class QuadcopterMinPoolNetwork(nn.Module):
    """
    obs = physical (13D) + depth image (160x120x1)
    physical -> MLP
    depth image -> CNN
    """
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        print("QuadcopterNetwork init ...")

        actions_num = kwargs.pop('actions_num', 4)
        input_shape = kwargs.pop('input_shape', (13+160*120,))

        self.central_value = params.get('central_value', False)
        self.value_size = params.get('value_size', 1)

        self.physical_mlp = nn.Sequential(
            nn.Linear(13, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 20), 
            nn.ReLU(), 
        )
        self.depth_min_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=40, stride=40), 
            nn.Flatten(), 
        )
        self.mean_linear = nn.Sequential(
            nn.Linear(20+12, 16), 
            nn.Linear(16, actions_num)
        )
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), 
            requires_grad=True
        )
        self.sigma_act = nn.ReLU()
        self.value_linear = nn.Sequential(
            nn.Linear(20+12, 16), 
            nn.Linear(16, 1)
        )

    def is_rnn(self):
        return False

    def forward(self, obs):
        batch_size = obs['obs'].shape[0]
        physical_obs = obs['obs'][..., 0:13]
        # from 1d to (1, 1, 120, 160)
        depth_image = obs['obs'][..., 13:].reshape(batch_size, 1, 120, 160)

        physical_out = self.physical_mlp(physical_obs)
        depth_image_out = -self.depth_min_pool(-depth_image)

        mean_output = self.mean_linear(torch.cat([physical_out, depth_image_out], dim=1))
        sigma = mean_output * 0.0 + self.sigma_act(self.sigma)
        value_output = self.value_linear(torch.cat([physical_out, depth_image_out], dim=1))
        if self.central_value:
            return value_output, None
        return mean_output, sigma, value_output, None


from rl_games.algos_torch.network_builder import NetworkBuilder

class QuadcopterMinPoolNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
    
    def build(self, name, **kwargs):
        return QuadcopterMinPoolNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)