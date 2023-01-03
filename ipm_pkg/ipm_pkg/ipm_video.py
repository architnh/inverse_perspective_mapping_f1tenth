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
    capture_images = True

    folder_name = "ipm_images"
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cam_jsons = CAM_JSONS
    cam_nums = CAM_NUMS

    cam_jsons_paths = []
    for cam_json in cam_jsons:
        cam_jsons_paths.append(os.path.join("calibration", "complete_calibrations", cam_json))

    if sim:
        img_paths = ['ipm_images/f_3.jpg', 'ipm_images/b_3.jpg', 'ipm_images/l_3.jpg', 'ipm_images/r_3.jpg']
        car = Car(cam_jsons_paths, cam_nums, debug=True, simulation=True, sim_image_paths=img_paths)
    else:
        car = Car(cam_jsons_paths, cam_nums, debug=True)

    # Create a plane in the car frame
    plane = Plane(IPM_PLANE_X_SIZE, IPM_PLANE_Y_SIZE, METER_PER_PIX)
    xyz = plane.xyz_coord(flat=True)
    image_center = np.array((plane.W / 2, plane.H / 2), dtype=np.int32)

    cam_pix_coords = car.project_points(xyz, plane.H, plane.W, with_distortion=True)

    distance_past = 0 #initialise
    head_past = 0 #initialize
    t_past = 0
    seen_tag = False
    seen_tag_twice = False
    vx_history = []
    vy_history = []

    #image_captures
    cnt = 0
    t_capture = 0
    while True:
        # Get the frames
        images = car.get_images()

        # Interpolate images
        ipm_images = car.interpolate_images(images, cam_pix_coords)

        # Look for apriltags in all images
        tag_uv, tag_cam_num = car.find_apriltags(images, plane)
        t_current = time.time()

        if tag_uv is not None:
            distance_current, head_current = pixel_dist_and_heading(image_center, tag_uv, METER_PER_PIX)
            if seen_tag:  # If a tag has been seen in the previous frame
                delta_t = t_current - t_past
                vx, vy = estimate_velocity(distance_past, distance_current, head_past, head_current, delta_t)

                # Process the velocity
                vx_history.append(vx)
                vy_history.append(vy)
                if vx_history.__len__() > 3:
                    vx_history.pop(0)
                if vy_history.__len__() > 3:
                    vy_history.pop(0)
                vx_avg = np.mean(vx_history)
                vy_avg = np.mean(vy_history)
                if abs(vx_avg) < 0.1:
                    vx_avg = 0
                if abs(vy_avg) < 0.1:
                    vy_avg = 0

                seen_tag_twice = True

            else: 
                seen_tag = True
            distance_past = distance_current
            head_past = head_current
            t_past = t_current

        if time.time() - t_past > .8:
            # If we haven't seen a tag in more than 0.8 seconds, reset
            seen_tag = False
            seen_tag_twice = False
            vx_history = []
            vx_history = []

        # Show the images
        combined_image = ipm_images[0] + ipm_images[1] + ipm_images[2] + ipm_images[3]
        combined_image = car.draw_car_and_cameras(combined_image)
        if seen_tag_twice and tag_uv is not None:
            combined_image = draw_point_uv(combined_image, tag_uv, RED, size=5)
            cv2.putText(combined_image, f'Distance : {round(distance_current, 2)} m', (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Heading : {np.round(head_current, 2)} rad', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Velocity X : {np.round(vx_avg, 2)} m/s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Velocity Y : {np.round(vy_avg, 2)} m/s', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)

        cv2.imshow("frame", combined_image)
        k = cv2.waitKey(1)

        if capture_images:
            if t_current - t_capture > 8:

                cv2.imwrite(os.path.join(folder_path, "{}_good_image.jpg".format(cnt)), combined_image)
                cv2.imwrite(os.path.join(folder_path, "{}_f.jpg".format(cnt)), images[0])
                cv2.imwrite(os.path.join(folder_path, "{}_b.jpg".format(cnt)), images[1])
                cv2.imwrite(os.path.join(folder_path, "{}_l.jpg".format(cnt)), images[2])
                cv2.imwrite(os.path.join(folder_path, "{}_r.jpg".format(cnt)), images[3])
                print("captured images")
                t_capture = t_current
                cnt += 1

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    car.release_cameras()
    cv2.destroyAllWindows()
