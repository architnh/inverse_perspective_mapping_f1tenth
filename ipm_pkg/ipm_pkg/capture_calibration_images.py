import cv2
import os
import time

"""
This script is used to capture calibration images for the camera calibration on the car. 
Use space to capture an image and then press esc to quit
Lenny Davis
"""
class CalibrationCapture:
    def __init__(self, camera_number, settings=None):
        self.camera_number = camera_number
        if settings is None:
            self.cam = cv2.VideoCapture(camera_number)
            self.cam.set(3, 640)
            self.cam.set(4, 480)
            self.cam.set(cv2.CAP_PROP_FPS, 30)

        else:
            self.cam = cv2.VideoCapture(settings)

        if self.cam.isOpened():
            print("Camera open")
        else:
            print("Camera open failed")

    def getimage(self):
        ret, frame = self.cam.read()
        return frame


if "__main__" == __name__:
    # Inputs
    folder_name = "camera_f_lowres"
    define_params = False
    cam_num = 2

    folder_path = os.path.join(os.getcwd(), "calibration", "calibration_images", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if define_params:
        settings_str = "v4l2src device=/dev/video" + str(cam_num) + " extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw, format=BGR ! appsink"
        cam = CalibrationCapture(cam_num, settings=settings_str)
    else:
        cam = CalibrationCapture(cam_num)
    img_counter = 0
    print("Press space to capture an image")
    print("Press esc to quit")
    delta = 0
    previous = 0
    while True:
        frame = cam.getimage()
        current = time.time()
        delta += current - previous
        previous = current
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if delta > 2:
            # time passed
            img_name = "image_{}.jpg".format(img_counter)
            cv2.imwrite(os.path.join(folder_path, img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
            delta = 0

cam.cam.release()
cv2.destroyAllWindows()
