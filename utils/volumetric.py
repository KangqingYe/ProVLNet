import numpy as np
import torch
class Cuboid3D:
    def __init__(self, position, sides):
        self.position = position
        self.sides = sides

def get_rotation_matrix(axis, theta):
    """Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_coord_volume(coord_volume, theta, axis):
    shape = coord_volume.shape
    device = coord_volume.device

    rot = get_rotation_matrix(axis, theta)
    rot = torch.from_numpy(rot).type(torch.float).to(device)

    coord_volume = coord_volume.view(-1, 3)
    coord_volume = rot.mm(coord_volume.t()).t()

    coord_volume = coord_volume.view(*shape)

    return coord_volume