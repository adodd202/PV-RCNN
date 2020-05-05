import numpy as np

def points_in_convex_polygon(points, polygon, ccw=True):
    """points (N, 2) | polygon (M, V, 2) | mask (N, M)"""
    polygon_roll = np.roll(polygon, shift=1, axis=1)
    polygon_side = (-1) ** ccw * (polygon - polygon_roll)[None]
    vertex_to_point = polygon[None] - points[:, None, None]
    mask = (np.cross(polygon_side, vertex_to_point) > 0).all(2)
    return mask


def box3d_to_bev_corners(boxes):
    """
    :boxes np.ndarray shape (N, 7)
    :corners np.ndarray shape (N, 4, 2) (ccw)
    """
    xy, _, wl, _, yaw = np.split(boxes, [2, 3, 5, 6], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
    corners = 0.5 * np.r_[-1, -1, +1, -1, +1, +1, -1, +1]
    corners = (wl[:, None] * corners.reshape(4, 2))
    corners = np.einsum('ijk,imk->imj', R, corners) + xy[:, None]
    return corners


class PointsInCuboids:
    """Takes ~10ms for each scene."""

    def __init__(self, points):
        self.points = points

    def _height_threshold(self, boxes):
        """Filter to z slice."""
        z1 = self.points[:, None, 2]
        z2, h = boxes[:, [2, 5]].T
        mask = (z1 > z2 - h / 2) & (z1 < z2 + h / 2)
        return mask

    def _get_mask(self, boxes):
        polygons = box3d_to_bev_corners(boxes)
        mask = self._height_threshold(boxes)
        mask &= points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return list of points in each box."""
        mask = self._get_mask(boxes).T
        points = list(map(self.points.__getitem__, mask))
        return points


class PointsNotInRectangles(PointsInCuboids):

    def _get_mask(self, boxes):
        polygons = box3d_to_bev_corners(boxes)
        mask = points_in_convex_polygon(
            self.points[:, :2], polygons)
        return mask

    def __call__(self, boxes):
        """Return array of points not in any box."""
        mask = ~self._get_mask(boxes).any(1)
        return self.points[mask]


def points_in_boxes(points, boxes):
    """
    Can use this approach: 
    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    For numpy & torch implementations:
    https://towardsdatascience.com/speed-up-your-python-code-with-broadcasting-and-pytorch-64fbd31b359
    boxes      - Tensor of shape [b, 7]
    keypoints  - Tensor of shape [2048, 3]
    Output     - Tensor of shape [2048], boolean
    """

    # Unpack tensors
    points = points.cpu().numpy()
    boxes = boxes.cpu().numpy()
    (x, y, z, w, l, h, theta) = np.split(boxes, 7, axis=1)
    q = np.full(l.shape, 2)

    # Calculate what the cuboid coordinates are
    Ax = -np.multiply(l/q, np.sin(theta)) + np.multiply(w/q, np.cos(theta))
    Ay = -np.multiply(l/q, np.cos(theta)) - np.multiply(w/q, np.sin(theta))
    Az = -h/q
    A = np.concatenate([Ax, Ay, Az], axis=1)

    Bx =  np.multiply(l/q, np.sin(theta)) + np.multiply(w/q, np.cos(theta))
    By =  np.multiply(l/q, np.cos(theta)) - np.multiply(w/q, np.sin(theta))
    Bz = -h/q
    B = np.concatenate([Bx, By, Bz], axis=1)

    Dx = -np.multiply(l/q, np.sin(theta)) - np.multiply(w/q, np.cos(theta))
    Dy = -np.multiply(l/q, np.cos(theta)) + np.multiply(w/q, np.sin(theta))
    Dz = -h/q
    D = np.concatenate([Dx, Dy, Dz], axis=1)

    Ex = -np.multiply(l/q, np.sin(theta)) + np.multiply(w/q, np.cos(theta))
    Ey = -np.multiply(l/q, np.cos(theta)) - np.multiply(w/q, np.sin(theta))
    Ez =  h/q
    E = np.concatenate([Ex, Ey, Ez], axis=1)

    # Centers
    cen = np.concatenate([x,y,z], axis=1)
    A, B, D, E = A + cen, B + cen, D + cen, E + cen

    # Add another dimension for broadcasting
    A = np.expand_dims(A, axis=1)
    B = np.expand_dims(B, axis=1)
    D = np.expand_dims(D, axis=1)
    E = np.expand_dims(E, axis=1)

    # Compute vectors of cuboid
    AM = (points[None, ...] - A)  # (N, M, 3)
    AB = (B - A)                  # (N, 1, 3)
    AD = (D - A)                  # (N, 1, 3) 
    AE = (E - A)                  # (N, 1, 3) 

    # Dot products of cuboid vectors and the points to check inside
    AM_AB = np.sum(AM * AB, axis=-1)  # (N, M)
    AB_AB = np.sum(AB * AB, axis=-1)  # (N, 1)
    AM_AD = np.sum(AM * AD, axis=-1)  # (N, M)
    AD_AD = np.sum(AD * AD, axis=-1)  # (N, 1)
    AM_AE = np.sum(AM * AE, axis=-1)  # (N, M)
    AE_AE = np.sum(AE * AE, axis=-1)  # (N, 1)

    # Create conditionals
    cond0 = (0 < AM_AB) & (AM_AB < AB_AB)  # (N, M)
    cond1 = (0 < AM_AD) & (AM_AD < AD_AD)  # (N, M)
    cond2 = (0 < AM_AE) & (AM_AE < AE_AE)  # (N, M)

    # If all conditionals are met, the point is in the box
    in_box = cond0 & cond1 & cond2  # (N, M) = in_box.shape
    w = np.any(in_box, axis=0)
    return w