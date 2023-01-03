from constants import *
import cv2
import numpy as np


def draw_camera(camera, image, color=WHITE):
    img_h = image.shape[0]
    img_w = image.shape[1]

    camera_pos = camera.return_position()
    camera_pos = camera_pos / METER_PER_PIX

    arrow_start = int(camera_pos[0] + img_w / 2), int(-camera_pos[1] + img_h / 2)
    arrow_end = int((camera_pos[0] + img_w / 2) + 20 * np.cos(camera.yaw)), int((-camera_pos[1] + img_h/ 2) - 20 * np.sin(camera.yaw))

    image = cv2.arrowedLine(image, arrow_start, arrow_end, color, 2, tipLength=0.6)
    image = cv2.circle(image, arrow_start, 3, color, -1)
    return image


def draw_car(image, color=PURPLE):
    img_h = image.shape[0]
    img_w = image.shape[1]
    
    car_h = CAR_WIDTH / METER_PER_PIX
    car_w = CAR_LENGTH / METER_PER_PIX
    image_center = (int(img_w/2), int(img_h/2))

    car_upper_left_x = int(img_w/ 2 - car_w)
    car_upper_left_y = int(img_h/ 2 - car_h / 2)

    car_lower_right_x = int(img_w / 2)
    car_lower_right_y = int(img_h / 2 + car_h / 2)
    
    image = cv2.rectangle(image, (car_upper_left_x, car_upper_left_y),
                                   (car_lower_right_x, car_lower_right_y), (255, 0, 255), -1)
    # Draw coordinate arrows
    image = cv2.arrowedLine(image, image_center, (image_center[0] + 25, image_center[1]), RED, 2)
    image = cv2.arrowedLine(image, image_center, (image_center[0], image_center[1] - 25), GREEN, 2)

    return image


def draw_point_uv(image, uv, color=GREEN, size=25):
    pix_u = int(uv[0])
    pix_v = int(uv[1])

    image = cv2.circle(image, (pix_u, pix_v), size, color, -1)
    return image

def draw_point_xyz(image, xyz, color=GREEN):
    img_h = image.shape[0]
    img_w = image.shape[1]

    pix_u = int(img_w / 2 + (xyz[0] / METER_PER_PIX))
    pix_v = int(img_h / 2 - (xyz[1] / METER_PER_PIX))

    image = cv2.circle(image, (pix_u, pix_v), 2, color, -1)


    return image




