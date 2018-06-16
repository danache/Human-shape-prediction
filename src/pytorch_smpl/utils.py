""" Util functions for SMPL
@@batch_skew
@@batch_rodrigues
@@batch_lrotmin
@@batch_global_rigid_transformation
"""
import numpy as np
import torch
import torch.nn.functional as F


def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape[0]

    a1 = -vec[:, 2]
    a2 = vec[:, 1]
    a3 = vec[:, 2]

    a5 = -vec[:, 0]
    a6 = -vec[:, 1]
    a7 = vec[:, 0]

    a0 = torch.zeros_like(a1).float().cuda()
    a4 = torch.zeros_like(a1).float().cuda()
    a8 = torch.zeros_like(a1).float().cuda()

    skew_mat = torch.cat([a0, a1, a2, a3, a4, a5, a6, a7, a8], 1)
    skew_mat = skew_mat.view(batch_size, 3, 3)

    return skew_mat



def batch_rodrigues(theta):
    """
    Theta is N x 3
    """
    batch_size = theta.shape[0]
    angle = torch.unsqueeze((theta + 1e-8).norm(dim=1), -1)
    r = torch.unsqueeze(theta / angle, -1)

    angle = torch.unsqueeze(angle, -1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    outer = torch.bmm(r, r.transpose(1, 2))

    eyes = torch.unsqueeze(torch.eye(3), 0).repeat(batch_size, 1, 1).cuda()
    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size)
    return R


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """

    N = Rs.shape[0]
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = torch.from_numpy(np.asarray(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]])).float().cuda()
        rot_x = rot_x.repeat(N, 1).view(N, 3, 3)
        root_rotation = torch.bmm(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t, name=None):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = F.pad(R, (0, 0, 0, 1, 0, 0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).cuda()], 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.bmm(results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js, torch.zeros([N, 24, 1, 1]).cuda()], 2)


    init_bone = torch.bmm(results.view(-1, 4, 4), Js_w0.view(-1, 4, 1))
    init_bone = init_bone.view(N, 24, 4, 1)

    # Append empty 4 x 3:
    init_bone = F.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0))
    A = results - init_bone

    return new_J, A
