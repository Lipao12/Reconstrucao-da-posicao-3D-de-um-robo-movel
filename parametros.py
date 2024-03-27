import numpy as np
import json

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

def extrinsic_homogeneous(R, T):
    M = np.hstack((R, T))
    M = np.vstack((M, [0, 0, 0, 1]))
    return M

def extrinsic_homogeneous_inv(R, T):
    M = extrinsic_homogeneous(R, T)
    return np.linalg.inv(M)