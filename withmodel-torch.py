import cv2
from ultralytics import YOLO
from djitellopy import Tello

# Muat model YOLOv8
model = YOLO('best.pt')

# Inisialisasi koneksi ke drone Tello
drone = Tello()
drone.connect()
drone.streamon()

# Cek baterai drone (opsional)
print(f"Battery: {drone.get_battery()}%")

try:
    while True:
        # Ambil frame dari stream Tello
        frame = drone.get_frame_read().frame

        # Jalankan deteksi pada frame
        results = model.predict(frame, conf=0.5)

        # Tampilkan frame yang sudah diberi anotasi
        annotated_frame = results[0].plot()

        # Tampilkan hasil di jendela OpenCV
        cv2.imshow('Drone Stream - YOLOv8', annotated_frame)

        # Tekan 'q' untuk keluar dari jendela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Hentikan stream dan matikan koneksi dasdwrone
    drone.streamoff()
    drone.end()

# Tutup jendela
cv2.destroyAllWindows()
