import numpy as np
import torch
import torch.nn as nn


class Camera(nn.Module):

    def __init__(self):
        super(Camera, self).__init__()



    def forward(self, joints, batch_size):
        self.tx = torch.zeros((1, batch_size, 1, 1))
        self.ty = torch.zeros((1, batch_size, 1, 1))
        self.tz = torch.rand((1, batch_size, 1, 1)) * 10


        t = torch.cat([self.tx, self.ty, self.tz], 3)
        camera_joints = joints + t.float().cuda()

        # Assertions
        diff = (camera_joints - joints).detach().cpu().numpy()
        diff = (diff[0, :, 0, 2] - np.squeeze(self.tz.numpy()))
        assert (diff < 1e-5).all()

        x = camera_joints[:, :, :, 0]
        y = camera_joints[:, :, :, 1]
        z = camera_joints[:, :, :, 2]

        u = (x / (z + 1e-8))
        v = (y / (z + 1e-8))

        u = u - torch.unsqueeze(torch.mean(u, 2), -1)
        v = v - torch.unsqueeze(torch.mean(v, 2), -1)

        projection = torch.stack([u, v], -1)
        return projection






