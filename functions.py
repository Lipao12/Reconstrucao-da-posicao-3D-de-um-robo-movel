import numpy as np
def homogenous_point(image_points):
    pts0 = image_points[0]
    pts0 = pts0.T
    pts0 = np.vstack((pts0, np.ones(pts0.shape[1])))
    pts1 = image_points[1]
    pts1 = pts1.T
    pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2 = image_points[2]
    pts2 = pts2.T
    pts2 = np.vstack((pts2, np.ones(pts2.shape[1])))
    pts3 = image_points[3]
    pts3 = pts3.T
    pts3 = np.vstack((pts3, np.ones(pts3.shape[1])))

    return pts0, pts1, pts2, pts3


def triangulate_point(pts0, pts1, pts2, pts3, P0, P1, P2, P3):
    M = np.zeros((12, 8))
    M[:3, :4] = P0
    M[3:6, :4] = P1
    M[6:9, :4] = P2
    M[9:, :4] = P3
    M[:3, 4] = -pts0.flatten()
    M[3:6, 5] = -pts1.flatten()
    M[6:9, 6] = -pts2.flatten()
    M[9:, 7] = -pts3.flatten()

    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]

def triangulate(image_points, P0, P1, P2, P3):
    pts0, pts1, pts2, pts3 = homogenous_point(image_points)

    X = triangulate_point(pts0, pts1, pts2, pts3, P0, P1, P2, P3)
    X = np.array(X)
    X = X.reshape(-1, 1)
    return X


def homogenous_point_2Cam(image_points):
    pts1 = image_points[0]
    pts1 = pts1.T
    pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2 = image_points[1]
    pts2 = pts2.T
    pts2 = np.vstack((pts2, np.ones(pts2.shape[1])))

    return pts1, pts2


def triangulate_point_2Cam(pts1, pts2, P1, P2):
    M = np.zeros((6, 6))
    M[:3,:4] = P1
    M[3:, :4] = P2
    M[:3,4] = -pts1.flatten()
    M[3:,5] = -pts2.flatten()

    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]

def triangulate_2Cam(image_points, P1, P2):
    pts1, pts2 = homogenous_point_2Cam(image_points)

    X = triangulate_point_2Cam(pts1, pts2, P1, P2)
    X = np.array(X)
    X = X.reshape(-1, 1)
    return X


def homogenous_point_3Cam(image_points):
    pts1 = image_points[0]
    pts1 = pts1.T
    pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2 = image_points[1]
    pts2 = pts2.T
    pts2 = np.vstack((pts2, np.ones(pts2.shape[1])))
    pts3 = image_points[2]
    pts3 = pts3.T
    pts3 = np.vstack((pts3, np.ones(pts3.shape[1])))

    return pts1, pts2, pts3


def triangulate_point_3Cam(pts0, pts1, pts2, P0, P1, P2):
    M = np.zeros((9, 7))
    M[:3, :4] = P0
    M[3:6, :4] = P1
    M[6:9, :4] = P2
    M[:3, 4] = -pts0.flatten()
    M[3:6, 5] = -pts1.flatten()
    M[6:9, 6] = -pts2.flatten()

    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]

def triangulate_3Cam(image_points, P0, P1, P2):
    pts0, pts1, pts2 = homogenous_point_3Cam(image_points)

    X = triangulate_point_3Cam(pts0, pts1, pts2, P0, P1, P2)
    X = np.array(X)
    X = X.reshape(-1, 1)
    return X





def distance_3d(robot, cam):
    x1, y1, z1, _ = robot
    x2, y2, z2, _= cam
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

