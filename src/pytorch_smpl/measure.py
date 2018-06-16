import torch


def compute_volume(verts, faces):

    verts = torch.transpose(verts, 0, 2)
    faces = faces.view(-1)

    px = verts[0][faces]
    px = px.view((13776, 3, -1))

    py = verts[1][faces]
    py = py.view((13776, 3, -1))

    pz = verts[2][faces]
    pz = pz.view((13776, 3, -1))

    triangle = torch.stack([px, py, pz], 2)
    triangle = triangle.permute(2,1,0,3)

    v321 = triangle[2][0] * triangle[1][1] * triangle[0][2]
    v231 = triangle[1][0] * triangle[2][1] * triangle[0][2]
    v312 = triangle[2][0] * triangle[0][1] * triangle[1][2]
    v132 = triangle[0][0] * triangle[2][1] * triangle[1][2]
    v213 = triangle[1][0] * triangle[0][1] * triangle[2][2]
    v123 = triangle[0][0] * triangle[1][1] * triangle[2][2]

    total_volume = (-v321 + v231 + v312 - v132 - v213 + v123) / 6.0

    return torch.sum(total_volume, 0)

def compute_height(verts):
    verts = verts.permute(1,2,0)
    max = torch.max(torch.abs(verts[411] - verts[3464]), 0)
    return max[0]
