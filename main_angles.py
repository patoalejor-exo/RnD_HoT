import numpy as np
import math
import cv2
import glob
import os
import matplotlib.pyplot as plt

def compute_angle_2D(p1, p2, p3):
    # Calculate the distances between the points
    a = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
    b = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    c = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)

    # Apply the law of cosines
    angle = math.acos((b**2 + a**2 - c**2) / ((2*a*b)+1e-6))

    # Convert the angle to degrees
    angle = math.degrees(angle)

    return angle


def show2Dpose(kps, img):
    colors = [(138, 201, 38),
              (25, 130, 196),
              (255, 202, 58)]

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    angles_connection = [[1, 2, 3],  # leg right
                         [4, 5, 6],  # leg left
                         [0, 1, 2],  # hip right
                         [0, 4, 5],  # hip left
                         [8, 14, 16],  # arm right
                         [8, 11, 13],  # arm left
                         [3, 0, 10],  # toe to head
                         ]
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)  # White color
    line_type = 2

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]),
                 (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-
                   1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-
                   1, color=colors[LR[j]-1], radius=3)
        text_position = start
        cv2.putText(img, f'{c[0]}', text_position, font,
                    font_scale, text_color, line_type)
        text_position = end
        cv2.putText(img, f'{c[1]}', text_position, font,
                    font_scale, text_color, line_type)

    for jj, cc in enumerate(angles_connection):
        start = map(int, kps[cc[0]])
        center = map(int, kps[cc[1]])
        end = map(int, kps[cc[2]])

        start = list(start)
        center = list(center)
        end = list(end)

        start = [start[0] - center[0], start[1] - center[1]]
        end = [end[0] - center[0], end[1] - center[1]]
        center = [0, 0]

        # print(f" --- {cc} --- ")
        # print(f"{start = }")
        # print(f"{center = }")
        # print(f"{end = }")

        # Define the position where the text will start
        text_position = map(int, kps[3])
        text_position = list(text_position)
        text_position[0] = 25
        text_position[1] = 25 + jj * 25
        angle = compute_angle_2D(start, center, end)
        cv2.putText(img, f'{cc} = {angle:.2f}     ',
                    text_position, font, 0.5, text_color, 1)

    return img


def main():

    
    all_keypoints = glob.glob('./demo/output/*walk*_mid/**/input_keypoints_2d.npz', recursive=True)
    print(all_keypoints)
    for path_to_keypoints in all_keypoints:
        print(f"Reading {path_to_keypoints}")
        keypoints = np.load(path_to_keypoints, allow_pickle=True)['reconstruction']
        print(f"{keypoints.shape = }")
        _lbl = path_to_keypoints.split(os.sep)[1]
    
        path_to_video = path_to_keypoints.replace('.npz', f'_{_lbl}_angles.mp4')
        # Create a figure and a keypoints
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1'
        out = cv2.VideoWriter(path_to_video, fourcc, 24.0, (640, 480))

        for i in range(keypoints.shape[1]):
            print(f' === Frame {i} === ')
            empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
            kps = keypoints[0, i, :, :]
            img = show2Dpose(kps, empty_image)
            cv2.imshow('frame', img)
            out.write(img)
            if cv2.waitKey(2) == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
