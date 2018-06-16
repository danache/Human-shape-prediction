import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Camera(nn.Module):

    def __init__(self):
        super(Camera, self).__init__()

    # TODO reuse src in "init_camera" and "init_camera_randomly"
    def _init_camera(self, camera_parameters):
        self.tx = Parameter(camera_parameters[:, 0].view(camera_parameters.shape[0], 1, 1))
        self.ty = Parameter(camera_parameters[:, 1].view(camera_parameters.shape[0], 1, 1))
        self.tz = Parameter(camera_parameters[:, 2].view(camera_parameters.shape[0], 1, 1))
        self.camera_parameters = camera_parameters

    def _init_camera_randomly(self, batch_size):
        self.tx = Parameter(torch.from_numpy(np.asarray(np.zeros((batch_size, 1, 1))))).float().cuda() + 9
        self.ty = Parameter(torch.from_numpy(np.asarray(np.zeros((batch_size, 1, 1))))).float().cuda() + 9
        self.tz = Parameter(torch.from_numpy(np.asarray(np.random.uniform(5, 5, (batch_size, 1, 1))))).float().cuda()
        self.camera_parameters = torch.squeeze(torch.cat([self.tx, self.ty, self.tz], 1))


    def forward(self, joints):
        t = torch.cat([self.tx, self.ty, self.tz], 2)
        joints = joints + t.float().cuda()

        x = joints[:,:,0]
        y = joints[:,:,1]
        z = joints[:,:,2]

        u = (x / (z + 1e-8))
        v = (y / (z + 1e-8))

        u = u - torch.unsqueeze(torch.mean(u, 1), -1)
        v = v - torch.unsqueeze(torch.mean(v, 1), -1)

        projection = torch.stack([u, v], 1)
        return projection






