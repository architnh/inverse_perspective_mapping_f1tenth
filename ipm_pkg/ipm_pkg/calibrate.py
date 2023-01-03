import cv2
import numpy as np
import os
import glob
import json
from datetime import datetime
import matplotlib.pyplot as plt

def calibrate_camera(json_name, checkerboard, checksize, display=False, debug=True, write_files=True):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2) * checksize

    # Extracting path of individual image stored in a given directory
    images = glob.glob(os.path.join(os.getcwd(), "calibration", "calibration_images" , json_name + "/*.jpg"))

    if debug:
        print(f"Found {len(images)} images, detecting corners")

    for i in range(len(images)):
        fname = images[i]
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if display:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)
            if debug:
                print(f"[{i + 1}/{len(images)}] Chessboard found")


        else:
            if debug:
                print(f"[{i + 1}/{len(images)}] FAILED - {os.path.relpath(fname)}")

    cv2.destroyAllWindows()
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistort and show
    img_dist = cv2.undistort(img, mtx, dist, None, mtx)
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axarr[1].imshow(cv2.cvtColor(img_dist, cv2.COLOR_BGR2RGB))
    plt.show()

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print(f"Total error: {mean_error}")

    # Write to json file
    if write_files:
        data = {'mtx': mtx.tolist(), 'dist': dist.tolist()}
        with open(os.path.join('calibration', 'intrinsic_calibrations', json_name + '.json'), 'w') as outfile:
            json.dump(data, outfile)
        print("Wrote intrinsic file")

def convert_json(fname, xyz, euler):
    """
    Takes a json file with intrinsic matrix and adds in the extrinsic information measured from the car
    INPUTS:
        fname: name of the json file
        xyz: 3x np array of camera position [x,y,z] (m)
        euler: 3x np array of camera orientation [yaw,pitch,roll](in radians)
    """
    with open(os.path.join(os.getcwd(), 'calibration', 'intrinsic_calibrations', fname + '.json'), 'rt') as j:
        camera_dat = json.load(j)

    # Extract the intrisnic parameters
    K = np.array(camera_dat['mtx'])
    D = camera_dat['dist']

    intrinsic_dict = {'fx': K[0, 0], 'fy': K[1, 1], 'u0': K[0, 2], 'v0': K[1, 2]}
    extrin_dict = {'x': xyz[0], 'y': xyz[1], 'z': xyz[2], 'yaw': euler[0], 'pitch': euler[1], 'roll': euler[2]}
    data = {'intrinsic': intrinsic_dict, 'extrinsic': extrin_dict, 'dist': D}

    out_fname = os.path.join('calibration', 'complete_calibrations', fname + '_complete.json')
    with open(out_fname, 'w') as outfile:
        json.dump(data, outfile, cls=NpEncoder)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if "__main__" == __name__:
    # INPUTS
    json_name = 'camera_f_lowres'
    checkerboard = (6, 8)
    checksize = .027  # m

    calibrate_camera(json_name, checkerboard, checksize, debug=True, display=False, write_files=True)

    convert_json(json_name, np.array([0, 0, 0]), np.array([0, 0, 0]))


