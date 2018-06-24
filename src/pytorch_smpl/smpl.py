import cPickle as pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMPL(nn.Module):

    def __init__(self, pkl_path, joint_type='cocoplus'):
        '''
        :param pkl_path:  is the path to a SMPL model
        '''
        super(SMPL, self).__init__()

        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)
        # Mean template vertices
        self.v_template = Parameter(torch.from_numpy(undo_chumpy(dd['v_template'])).float()).cuda()

        # Faces
        self.f = Parameter(torch.from_numpy(np.asarray(undo_chumpy(dd['f']), np.int32))).long().cuda()

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = Parameter(torch.from_numpy(shapedir).float()).cuda()
        self.shapedirs = torch.unsqueeze(self.shapedirs, 0)

        # Regressor for joint locations given shape - 6890 x 24
        J_regressor =  dd['J_regressor'].T.todense()
        self.J_regressor = Parameter(torch.from_numpy(J_regressor)).float().cuda()

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Parameter(torch.from_numpy(posedirs).float()).cuda()

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Parameter(torch.from_numpy(undo_chumpy(dd['weights'])).float()).cuda()
        # This returns 19 keypoints: 6890 x 19
        self.joint_regressor = Parameter(torch.from_numpy(dd['cocoplus_regressor'].T.todense()).float()).cuda()
        if joint_type == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()


    def cuda(self, device=None):
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def forward(self, beta, theta, get_skin=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        num_batch = beta.shape[0]
        # 1. Add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890 x 3
        beta = torch.unsqueeze(beta, 0)
        v_shaped = torch.bmm(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template


        # 2. Infer shape-dependent joint locations.
        Jx = torch.mm(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.mm(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.mm(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], 2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3).cuda()).view(-1, 207)
        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = torch.mm(pose_feature, self.posedirs).view(
            -1, self.size[0], self.size[1]) + v_shaped

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, False)

        # 5. Do skinning:
        # W is N x 6890 x 24
        W = self.weights.repeat(num_batch, 1).view(num_batch, -1, 24)
        # (N x 6890 x 24) x (N x 24 x 16)
        T = torch.bmm(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones((num_batch, v_posed.shape[1], 1)).cuda()], 2)
        v_homo = torch.bmm(T.view(-1, 4, 4), torch.unsqueeze(v_posed_homo, -1).view(-1, 4, 1))
        v_homo = v_homo.view(num_batch, 6890, 4, 1)


        verts = v_homo[:, :, :3, 0]

        # Get cocoplus or lsp joints:
        joint_x = torch.mm(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.mm(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.mm(verts[:, :, 2], self.joint_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)


        if get_skin:
            return verts, joints, Rs
        else:
            return joints

