import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
import apriltag
from constants import *
from viz_utils import *
###############################################################################
#                                                                             #
#                               IPM   UTILITIES                               #
#                                                                             #
###############################################################################

class Plane:
    """
    Defines a plane in the world
    https://github.com/darylclimb/cvml_project/tree/master/projections/ipm
    """
    def __init__(self, world_x_size, world_y_size, meter_per_pix):
        """
        Args:
        :param origin: [x, y, z] np.array of the orgin of the plane on the ground (z should be zero)
        :param euler: [roll, pitch, yaw] np.array of the euler angles of the plane
        :param col: number of desired columns in the plane
        :param row: number of desired rows in the plane
        :param scale: scale of the plane in meters
        """
        self.x_size, self.y_size = world_x_size, world_y_size

        self.W, self.H = int(world_x_size / meter_per_pix), int(world_y_size / meter_per_pix)
        self.meter_per_pix = meter_per_pix

    def xyz_coord(self, flat=False, front_only=False):
        """
        Returns:
            Grid coordinate:
        """

        if front_only:
            x = np.linspace(0, self.x_size, self.W)
            y = np.linspace(self.y_size/2, -self.y_size/2, self.H)
        else:
            x = np.linspace(-self.x_size/2, self.x_size/2, self.W)
            y = np.linspace(self.y_size/2, -self.y_size/2, self.H)

        xx, yy = np.meshgrid(x, y)
        xyz = np.stack((xx, yy, np.zeros_like(xx), np.ones_like(xx)), axis=2)

        if flat:
            xyz = xyz.reshape((-1, 4)).T
        return xyz

    def xyz2uv(self, xyz):
        """
        Converts from world coordinates to plane image coordinates
        """
        uv = np.zeros((2, 1))
        uv[0] = int(self.W / 2 + (xyz[0] / METER_PER_PIX))
        uv[1] = int(self.H / 2 - (xyz[1] / METER_PER_PIX))

        return uv

def meshgrid(xmin, xmax, num_x, ymin, ymax, num_y, is_homogeneous=True):
    """
    Grid is parallel to z-axis
    Returns:
        array x,y,z,[1] coordinate   [3/4, num_x * num_y]
    """
    x = np.linspace(xmin, xmax, num_x)
    y = np.linspace(ymin, ymax, num_y)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    if is_homogeneous:
        coords = np.stack([x, y, z, np.ones_like(x)], axis=0)
    else:
        coords = np.stack([x, y, z], axis=0)
    return coords


class Car:
    def __init__(self, camera_jsons, cam_nums, debug=False, simulation=False, sim_image_paths=None, undistort=False):
        """
        Args:
        :param camera_jsons: list of camera json files [font, back, left, right]
        :param cam_nums: list of camera numbers [front, back, left, right]
        """
        self.cams = []
        self.caps = []
        self.simulation = simulation
        self.sim_image_paths = sim_image_paths
        self.undistort = undistort

        for i in range(4):
            self.cams.append(Camera(camera_jsons[i]))
            #settings_str = "v4l2src device=/dev/video" + str(
            #    cam_nums[i]) + " extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw, format=BGR ! appsink"
            if not self.simulation:
                #self.caps.append(cv2.VideoCapture(settings_str))
                cap = cv2.VideoCapture(cam_nums[i])
                cap.set(3, CAMERA_WIDTH)
                cap.set(4, CAMERA_HEIGHT)
                self.caps.append(cap)
                if debug:
                    cv2.imshow('Image', self.caps[i].read()[1])
                    cv2.waitKey(0)
                cv2.destroyAllWindows()

    def project_points(self, xyz, plane_h, plane_w, with_distortion=True):
        """
        Args:
        :param xyz: [x, y, z, 1] np.array of the points to project
        :param plane_h: height of the plane
        :param plane_w: width of the plane
        :return: list containing np array of projected points
        """
        cam_pix_coords = []

        for cam in self.cams:
            cam_pix_coords.append(cam.project_points(xyz, plane_h, plane_w, with_distortion))

        return cam_pix_coords

    def get_images(self):
        """
        Returns:
            list of images [font, back, left, right]
        """
        images = []
        if self.simulation:
            for i in range(4):
                images.append(cv2.imread(self.sim_image_paths[i]))
        else:
            for cap in self.caps:
                images.append(cap.read()[1])

        return images

    def interpolate_images(self, images, cam_pix_coords):
        """
        Args:
        :param images: list of images [font, back, left, right]
        :param cam_pix_coords: list of camera pixel coordinates [font, back, left, right]
        :return: list of interpolated images [font, back, left, right]
        """
        interpolated_images = []

        for i in range(4):
            interpolated_images.append(interpolate_image(images[i], cam_pix_coords[i]))

        return interpolated_images

    def draw_car_and_cameras(self, image):
        """
        Args:
        :param image: image to draw cameras on
        :return: image with cameras drawn on it
        """
        image = draw_car(image)
        for i in range(4):
            image = draw_camera(self.cams[i], image)

        return image

    def release_cameras(self):
        """
        Releases all cameras
        """
        for cap in self.caps:
            cap.release()

    def find_apriltags(self, images, plane):
        """
        Args:
        :param images: list of images [font, back, left, right]
        :param plane: plane to find apriltags in
        :return: single apriltag location
        :return: camera num that detected the tag
        :return:
        """
        tag = None
        for i in range(4):
            tag_found, tag_corners = return_apriltag_location(images[i], return_center=False)
            if tag_found:
                tag, valid = return_bounding_midpoint(tag_corners, self.cams[i], plane)
                break

        return tag, i




class Camera:
    def __init__(self, file_path, cam_num=None, settings_str=None):
        with open(file_path, 'rt') as j:
            camera_dat = json.load(j)

        # Get camera intrinsic parameters
        ex = camera_dat['extrinsic']
        self.pitch = ex['pitch']
        self.roll = ex['roll']
        self.yaw = ex['yaw']
        self.x = ex['x']  # translations
        self.y = ex['y']  # translations
        self.z = ex['z']  # translations

        # Get camera extrensic parameters
        intrin = camera_dat['intrinsic']
        self.fx = intrin['fx']
        self.fy = intrin['fy']
        self.u0 = intrin['u0']
        self.v0 = intrin['v0']

        self.dist = np.array(camera_dat['dist'])

        self.mtx = np.array([[self.fx, 0, self.u0],
                              [0, self.fy, self.v0],
                              [0, 0, 1]])

        # Get camera matrices
        self.intrinsic, self.extrinsic = self.return_intrinsic_and_extrinsic()
        self.projection_matrix = self.intrinsic @ self.extrinsic

        # Create opencv camera object
        if cam_num:
            if settings_str is None:
                self.cap = cv2.VideoCapture(cam_num)
            else:

                self.cap = cv2.VideoCapture(settings_str)

    def return_intrinsic_and_extrinsic(self):
        """
        Returns:
            intrinsic 3x3 matrix
            extrinsic 3x4 matrix
        """
        intrinsic = np.array([[self.fx, 0, self.u0, 0],
                              [0, self.fy, self.v0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        T_car2cam = np.array([[1, 0, 0, -self.x],
                              [0, 1, 0, -self.y],
                              [0, 0, 1, -self.z],
                              [0, 0, 0, 1]])

        R_car2cam = np.eye(4)
        R_car2cam[:3, :3] = R.from_euler('ZXY', [self.yaw, self.roll, self.pitch]).as_matrix().T

        R_change_axis = np.array([[0., -1., 0., 0.],
                              [0., 0., -1., 0.],
                              [1., 0., 0., 0.],
                              [0., 0., 0., 1.]])

        H_w2c = R_change_axis @ R_car2cam @ T_car2cam  # Extrinsics

        return intrinsic, H_w2c

    def return_position(self):
        """
        Returns:
            position of the camera
        """
        return np.array([self.x, self.y, self.z])

    def project_points(self, xyz, plane_h, plane_w, with_distortion=True):
        """
        Take in xyz points and project them to the image plane

        Args:
            xyz: [4, n]
            plane_h: height of the plane (in rows)
            plane_w: width of the plane (in cols)
            with_distortion: if true, apply distortion to the points
        Returns:
            projected points: [2, n]
        """

        if with_distortion:
            xyz_ = self.extrinsic @ xyz
            point_ = xyz_[:3, :]
            z_val = xyz_[2, :]
            pix_coor, _ = cv2.projectPoints(point_, np.zeros((3, 1)), np.zeros((3, 1)), self.mtx, self.dist)
            pix_coor = pix_coor.squeeze().T
        else:
            # Project points to image plane
            pix_coor = self.projection_matrix @ xyz
            z_val = pix_coor[2, :]
            pix_coor = pix_coor[:2, :] / z_val # Normalize by depth value

        # Remove points behind the camera
        pix_coor[:, z_val < 0] = -1

        # Reshape to fit the plane
        pix_coor = np.reshape(pix_coor, (2, plane_h, plane_w))
        pix_coor = np.transpose(pix_coor, (1, 2, 0))

        return pix_coor

    def return_inverse_projection(self, uv):
        """
        Takes a pixel and returns its location in the world frame
        Assuming that the pixel lies on the z=0 plane of the world
        """
        valid = True
        xyz = None

        # Parse inputs
        u = uv[0]
        v = uv[1]

        # Calulate pixel depth
        H_c2w = np.linalg.inv(self.extrinsic)
        val = H_c2w[2, 0:3] @ np.array([(u - self.u0)/self.fx, (v - self.v0)/self.fy, 1]).reshape(3, 1)
        dist = -H_c2w[2, 3]/val

        if dist < 0:
            valid = False
        else:
            x_c = (u - self.u0)/self.fx * dist
            y_c = (v - self.v0)/self.fy * dist
            z_c = dist
            xyz = H_c2w @ np.array([x_c[0], y_c[0], z_c[0], 1]).reshape(4, 1)

            xyz = xyz[0:3].squeeze()
        return xyz, valid

    def capture_image(self):
        """
        Capture an image from the camera and return it as a numpy array
        """
        ret, frame = self.cap.read()
        return ret, frame


def interpolate_image(image, points):
    image_h, image_w = image.shape[:2]

    # round points to nearest integer
    points = np.round(points).astype(int)
    p_x, p_y = np.split(points, 2, axis=2)

    # remove points outside of image bounds
    bad_pixel = np.logical_or(np.logical_or(p_x < 0, p_x >= image_w),
                              np.logical_or(p_y < 0, p_y >= image_h))

    p_x = np.clip(p_x, 0, image_w - 1)
    p_y = np.clip(p_y, 0, image_h - 1)

    out = image[p_y, p_x].squeeze()
    out[bad_pixel.squeeze(), :] = 0
    return out

###############################################################################
#                                                                             #
#                           LOCATION   UTILITIES                              #
#                                                                             #
###############################################################################


def pixel_dist_and_heading(car_location, pixel_location, meters_per_pixel):
    """
    Find the distance and heading of a given pixel in the car frame

    Args:
        car_location: [car_x, car_y] in pixels
        pixel_location: [x, y]
    Returns:
        distance: float
        heading: float (in radians)
    """
    car_location = np.array([car_location[0], car_location[1]])
    dist = np.linalg.norm(pixel_location.reshape(1, 2) - car_location.reshape(1, 2)) * meters_per_pixel
    heading = np.arctan2(car_location[1] - pixel_location[1],
                         pixel_location[0] - car_location[0])

    return dist, heading



###############################################################################
#                                                                             #
#                           APRILTAG UTILS                                    #
#                                                                             #
###############################################################################
def return_apriltag_location(image, return_center=False, debug=False):
    """
    Find the location of the apriltag in the image
    ONLY WORKS FOR ONE TAG IN THE IMAGE
    Args:
        image: numpy array
        return_center: bool, if True, return the center of the tag, if false return corners
        debug: bool, if True, print the tag location
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    # define the AprilTags detector options and then detect the AprilTags
    # in the input image
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)

    r = None
    out = None
    ret = False

    if debug:
        print("[INFO] {} total AprilTags detected".format(len(results)))

    if len(results) > 0:
        r = results[0]
        ret = True
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        if debug:
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the imagethe (x, y)-coordinate pairs to integers
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] tag family: {}".format(tagFamily))
            # show the output image after AprilTag detection
            cv2.imshow("Image", image)
            cv2.waitKey(0)

        if return_center:
            out = r.center
        else:
            out = r.corners

    return ret, out


def return_bounding_midpoint(tag_corners, cam, plane):
    """
    Return the lower midpoint of the bounding box
    Args:
        tag_corners: list of corners of the apriltag
        cam: camera object
        plane: plane object
    Returns:
        midpoint: (u, v) tuple
    """
    # Get the midpoint of the lowest two tag corners
    out = np.argsort(tag_corners, axis=0)
    close_points = tag_corners[out[:, 1], :][2:, :]

    midpoint = np.mean(close_points, axis=0)
    midpoint_xyz, valid = cam.return_inverse_projection(midpoint)
    if valid:
        midpoint_uv = plane.xyz2uv(midpoint_xyz)
    else:
        midpoint_uv = None

    return midpoint_uv, valid


def estimate_velocity(distance1, distance2, heading1, heading2, delta_t):
    """
    Args:
    :param distance1: initial distance estimate of opponent car
    :param distance2: latest distance estimate of opponent car
    :param heading: direction estimate of where the opponent car is headed
    :param delta_t: Time between two readings
    :return: velocity estimate
    :return: positive indicates opponent is faster, moving away
    :return: negative indicates, opponent is slower, ego_car is gaining on opponent
    """
    velocity_x = (distance2*np.cos(heading2) - distance1*np.cos(heading1))/delta_t
    velocity_y = (distance2*np.sin(heading2) - distance1*np.sin(heading1))/delta_t

    return velocity_x, velocity_y


