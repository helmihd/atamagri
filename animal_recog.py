import time
import cv2
import threading
from djitellopy import tello
from tkinter import *
from PIL import Image, ImageTk
from xbox_one_controller import XboxController
import tensorflow as tf
import numpy as np

class GameControllerGUI:
    def __init__(self):
        # Initialize Xbox controller
        self.xbox_controller = XboxController()

        # Initialize Tkinter window
        self.root = Tk()
        self.root.title("Tello Drone Control GUI with Xbox Game Controller")
        self.root.minsize(800, 600)

        # Initialize video stream capture label
        self.cap_lbl = Label(self.root)

        # Prepare drone object
        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()

        # Initialize variables
        self.frame = self.drone.get_frame_read()

        # RC control values
        self.rc_controls = [0, 0, 0, 0]

        # Load TensorFlow model
        self.sess, self.detection_graph, self.input_tensor, self.boxes, self.scores, self.classes, self.num_detections = self.load_model()

    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile("frozen_inference_graph.pb", "rb") as f:
                serialized_graph = f.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        sess = tf.compat.v1.Session(graph=detection_graph)
        input_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        scores = detection_graph.get_tensor_by_name("detection_scores:0")
        classes = detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = detection_graph.get_tensor_by_name("num_detections:0")

        return sess, detection_graph, input_tensor, boxes, scores, classes, num_detections

    def detect_objects(self, frame):
        resized_frame = cv2.resize(frame, (300, 300))
        input_frame = np.expand_dims(resized_frame, axis=0)

        (det_boxes, det_scores, det_classes, det_num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.input_tensor: input_frame}
        )

        h, w, _ = frame.shape
        for i in range(int(det_num_detections[0])):
            if det_scores[0][i] > 0.5:
                box = det_boxes[0][i]
                (ymin, xmin, ymax, xmax) = (box[0] * h, box[1] * w, box[2] * h, box[3] * w)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        return frame

    def takeoff_land(self):
        if self.drone.is_flying:
            threading.Thread(target=lambda: self.drone.land()).start()
        else:
            threading.Thread(target=lambda: self.drone.takeoff()).start()

    def update_joystick(self):
        try:
            joystick_values = self.xbox_controller.read()
            left_joystick_x = joystick_values[0]
            left_joystick_y = joystick_values[1]
            right_joystick_x = joystick_values[2]
            right_joystick_y = joystick_values[3]
            start_button = joystick_values[14]

            if start_button:
                self.takeoff_land()
                time.sleep(0.15)

            self.rc_controls[0] = right_joystick_x
            self.rc_controls[1] = right_joystick_y
            self.rc_controls[2] = left_joystick_y
            self.rc_controls[3] = left_joystick_x

            if self.rc_controls != [0, 0, 0, 0]:
                self.drone.send_rc_control(self.rc_controls[0], self.rc_controls[1], self.rc_controls[2], self.rc_controls[3])
            else:
                self.drone.send_rc_control(0, 0, 0, 0)

            self.root.after(50, self.update_joystick)
        except Exception as e:
            print(f"Error in joystick update: {e}")

    def run_app(self):
        try:
            self.cap_lbl.pack(anchor="center")
            self.video_stream()
            self.update_joystick()
            self.root.mainloop()
        except Exception as e:
            print(f"Error in running app: {e}")
        finally:
            self.cleanup()

    def video_stream(self):
        try:
            h, w = 480, 720
            frame = self.frame.frame
            frame = cv2.resize(frame, (w, h))

            # Apply object detection
            frame = self.detect_objects(frame)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)
            self.cap_lbl.after(5, self.video_stream)
        except Exception as e:
            print(f"Error in video stream: {e}")

    def cleanup(self):
        try:
            print("Cleaning up resources...")
            self.drone.end()
            self.root.quit()
            exit()
        except Exception as e:
            print(f"Error in cleanup: {e}")

if __name__ == "__main__":
    gui = GameControllerGUI()
    gui.run_app()
