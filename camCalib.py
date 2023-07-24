#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import yaml
import pickle
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera using a video of a chessboard or a sequence of images.')
    parser.add_argument('input', help='input video file or glob mask')
    parser.add_argument('out', help='output calibration yaml file')
    parser.add_argument('--debug-dir', help='path to directory where images with detected chessboard will be written',
                        default=None)
    parser.add_argument('-c', '--corners', help='output corners file', default=None)
    parser.add_argument('-fs', '--framestep', help='use every nth frame in the video', default=5, type=int)
    # parser.add_argument('--figure', help='saved visualization name', default=None)
    args = parser.parse_args()

    if '*' in args.input:
        source = glob(args.input)
    else:
        source = cv2.VideoCapture(0)
        #source = cv2.VideoCapture('rtsp://admin:L2F0A54D@192.168.0.4/cam/realmonitor?channel=1&subtype=00&authbasic=YWRtaW46TDI0NzYzMzg=')
    square_size = 2.5
    pattern_size = (7, 7)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    flags = 0
    flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    flags |= cv2.CALIB_CB_FAST_CHECK
    flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    obj_points = []
    img_points = []
    h, w = 0, 0
    i = -1
    while True:
        i += 1
        if isinstance(source, list):
            # glob
            if i == len(source):
                break
            img = cv2.imread(source[i])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if i % args.framestep != 0:
                continue

        print('Searching for chessboard in frame ' + str(i) + '...')
        print('Image Height       : ',img.shape[0])
        print('Image Width        : ',img.shape[1])
        cv2.imshow("im", img)
        cv2.waitKey(1) 

        if(i==1000):
            break
        #img = cv2.resize(img,(1920,1080))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size, flags=flags)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, '%04d.png' % i), img_chess)
        if not found:
            print ('not found')
            continue
        img_points.append(corners.reshape(1, -1, 2))
        obj_points.append(pattern_points.reshape(1, -1, 3))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print ('ok')

    if args.corners:
        with open(args.corners, 'wb') as fw:
            pickle.dump(img_points, fw)
            pickle.dump(obj_points, fw)
            pickle.dump((w, h), fw)
        
# load corners
#    with open('corners.pkl', 'rb') as fr:
#        img_points = pickle.load(fr)
#        obj_points = pickle.load(fr)
#        w, h = pickle.load(fr)

    print('\nPerforming calibration...')
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print ("RMS:", rms)
    print ("camera matrix:\n", camera_matrix)
    print ("distortion coefficients: ", dist_coefs.ravel())

    #double fovx,fovy,focalLength,principalPoint,aspectRatio;
        # fovx=0
        # fovy=0
        # focalLength=0
        # principalPoint=0
        # aspectRatio=0
        # sx = 22.3
        # sy = 14.9;
        # fovx,fovy,focalLength,principalPoint,aspectRatio=cv2.calibrationMatrixValues(camera_matrix,(640,480),sx,sy)
        # print("fovy :",fovy)
        # print("fovx : ",fovx)
        # print("focal lenth ",focalLength)
    # # fisheye calibration
    # rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    #     obj_points, img_points,
    #     (w, h), camera_matrix, np.array([0., 0., 0., 0.]),
    #     None, None,
    #     cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, (3, 1, 1e-6))
    # print "RMS:", rms
    # print "camera matrix:\n", camera_matrix
    # print "distortion coefficients: ", dist_coefs.ravel()

    calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist() }
    cv2.destroyAllWindows()
    with open(args.out, 'w') as fw:
        yaml.dump(calibration, fw)
