import time
import cv2
import threading
from djitellopy import Tello
from tkinter import *
from PIL import Image, ImageTk
from xbox_one_controller import XboxController
from ultralytics import YOLO

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
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()

        # Initialize variables
        self.frame = self.drone.get_frame_read()

        # RC control values
        self.rc_controls = [0, 0, 0, 0]

        # Load YOLO model
        self.model = YOLO('best.pt')  # Replace 'best.pt' with your YOLO model path

    def detect_objects(self, frame):
        # Run YOLOv8 detection
        results = self.model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()
        return annotated_frame

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
            self.drone.streamoff()
            self.drone.end()
            self.root.quit()
            exit()
        except Exception as e:
            print(f"Error in cleanup: {e}")

if __name__ == "__main__":
    gui = GameControllerGUI()
    gui.run_app()
