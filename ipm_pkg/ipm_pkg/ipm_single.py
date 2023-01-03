import numpy as np
import cv2
from utils import *
import matplotlib.pyplot as plt
import os
from viz_utils import *
from constants import *
import time

if __name__ == "__main__":
    debug = False
    sim = False
    capture_images = False

    folder_name = "ipm_images_single"
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cam_json = 'camera_f_lowres_complete.json'
    cam_json_path = os.path.join("calibration", "complete_calibrations", cam_json)
    cam_num = 2
    cam = Camera(cam_json_path)
    cap = cv2.VideoCapture(cam_num)

    # Create a plane in the car frame
    plane = Plane(IPM_PLANE_X_SIZE, IPM_PLANE_Y_SIZE, METER_PER_PIX)
    xyz = plane.xyz_coord(flat=True, front_only=False)
    image_center = np.array((plane.W / 2, plane.H / 2), dtype=np.int32)
    cam_pix_coords = cam.project_points(xyz, plane.H, plane.W)

    initial_distance = 0 #initialise
    initial_head = 0 #initialize
    seen_tag = False
    plot_point = False
    t_past = 0
    t_current = 0
    cnt = 0
    delta = 0
    previous = 0
    current = 0
    tag_uv = None
    t_capture = 0
    while True:
        current = time.time()
        delta += current - previous
        previous = current

        # Get the frames
        image = cap.read()[1]

        # Interpolate images
        ipm_image = interpolate_image(image, cam_pix_coords)

        # Look for apriltags in all images
        tag_found, tag_corners = return_apriltag_location(image, return_center=False)
        if tag_found:
            tag_uv, valid = return_bounding_midpoint(tag_corners, cam, plane)

        if tag_uv is not None:
            # List the point distance and heading
            t_current = time.time()
            if seen_tag: 
                dist, head = pixel_dist_and_heading(image_center, tag_uv, METER_PER_PIX)
                delta_t = t_current - t_past
                velocity_x = (dist * np.cos(head) - initial_distance * np.cos(initial_head)) / delta_t
                velocity_y = (dist * np.sin(head) - initial_distance * np.sin(initial_head)) / delta_t
                initial_distance = dist
                initial_head = head
                plot_point = True
            else: 
                seen_tag = True
            t_past = t_current
            

        else:
            seen_tag = False
            plot_point = False

        if plot_point:
            ipm_image = draw_point_uv(ipm_image, tag_uv, RED, size=5)
            cv2.putText(ipm_image, f'Distance : {round(dist, 2)} m', (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_image, f'Heading : {np.round(head, 2)} rad', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_image, f'Velocity X : {np.round(velocity_x, 2)} m/s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_image, f'Velocity Y : {np.round(velocity_y, 2)} m/s', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)

        cv2.imshow("frame", ipm_image)
        k = cv2.waitKey(1)

        if capture_images:
            if current - t_capture > 8:
                cv2.imwrite(os.path.join(folder_path, "{}_good_image.jpg".format(cnt)), combined_image)
                cv2.imwrite(os.path.join(folder_path, "{}_f.jpg".format(cnt)), images[0])

                print("captured images")
                t_capture = t_current
                cnt += 1


        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()
