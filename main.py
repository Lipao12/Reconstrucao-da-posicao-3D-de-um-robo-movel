import cv2
import numpy as np
from cv2 import aruco
from parametros import *
from functions import *
import matplotlib.pyplot as plt

video_files = [
    "assets/camera-00.mp4",
    "assets/camera-01.mp4",
    "assets/camera-02.mp4",
    "assets/camera-03.mp4"
]

marker_id_to_detect = 0
MARKER_SIZE = 30 #cm
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

videos = [cv2.VideoCapture(file) for file in video_files]

#Load cameras parameters
K0, R0, T0, res0, dis0 = camera_parameters('camera_calibration/0.json')
K1, R1, T1, res1, dis1 = camera_parameters('camera_calibration/1.json')
K2, R2, T2, res2, dis2 = camera_parameters('camera_calibration/2.json')
K3, R3, T3, res3, dis3 = camera_parameters('camera_calibration/3.json')
M0, M1, M2, M3 = extrinsic_homogeneous_inv(R0, T0), extrinsic_homogeneous_inv(R1, T1), extrinsic_homogeneous_inv(R2, T2), extrinsic_homogeneous_inv(R3, T3)


P0 = K0 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ M0 #extrinsic_homogeneous(R0, T0) # mundo --> camera
P1 = K1 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ M1 #extrinsic_homogeneous(R1, T1)
P2 = K2 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ M2 #extrinsic_homogeneous(R2, T2)
P3 = K3 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ M3 #extrinsic_homogeneous(R3, T3)

all_cameras_have_aruco = True # Para os vetores de pontos pegarem os mesmos pontos
trajectory = []

while True:
    frames = [video.read() for video in videos]
    if any(frame is None for _, frame in frames):
        break

    for _, frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for _, frame in frames]

    points_2d = [[] for _ in range(len(videos))]
    newP2d = []
    camIdx = []
    for i, gray_frame in enumerate(gray_frames):
        corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        if ids is not None and 0 in ids:
            index = np.where(ids == marker_id_to_detect)[0][0]
            corners_mean = np.mean(corners[index][0], axis=0)
            points_2d[i].append(corners_mean)

            x, y = corners_mean.astype(int)
            cv2.putText(frames[i][1], f"ID: {index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    i = 0
    for p in points_2d:
        if len(p) > 0:
            newP2d.append(p)
            camIdx.append(i)
        i = i + 1

    try:
        if len(newP2d) >= 2:
            P = [P0, P1, P2, P3]
            points_2d = np.array(newP2d)
            if len(newP2d) == 4:
                points_3d = triangulate(points_2d, P0, P1, P2, P3)
            elif len(newP2d) == 3:
                points_3d = triangulate_3Cam(points_2d, P[camIdx[0]], P[camIdx[1]], P[camIdx[2]])
            elif len(newP2d) == 2:
                points_2d = points_2d[0], points_2d[1]
                points_3d = triangulate_2Cam(points_2d, P[camIdx[0]], P[camIdx[1]])

            if len(trajectory) == 0:
                trajectory = points_3d
            else:
                trajectory = np.hstack((trajectory, points_3d))
    except ZeroDivisionError as error:
        print(error)

    for i, frame in enumerate(frames):
        cv2.imshow(f"Camera-0{i}", frame[1])

    if cv2.waitKey(1) == ord('q'):
        break


for video in videos:
    video.release()

cv2.destroyAllWindows()

trajectory = np.array(trajectory)
print(trajectory.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = trajectory[0, :]
y = trajectory[1, :]
z = trajectory[2, :]

ax.plot(x.flatten(), y.flatten(), z.flatten(), )
ax.scatter(x[-1].flatten(), y[-1].flatten(), z[-1].flatten(), c='red', marker='o') # Plotar o ponto na posicao final

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

min_value = min(np.min(x), np.min(y), np.min(z))
max_value = max(np.max(x), np.max(y), np.max(z))
ax.set_xlim(min_value, max_value)
ax.set_ylim(min_value, max_value)
ax.set_zlim(0.55, 0.75)

plt.show()

