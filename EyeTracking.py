"""
Author: Larynt Sawfa Kenanga
"""
import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Cari itik pupil pada citra mata
def detect_pupil(eye):
    # Konversi citra ke grayscale
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    # Ambil tepi dengan Canny
    edged = cv2.Canny(gray, 30, 200)
    # Temukan lingkaran terbesar dengan HoughCircles
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=30, param2=45, minRadius=0, maxRadius=0)
    # Jika lingkaran ditemukan, kembalikan pusatnya
    if circles is not None:
        return np.round(circles[0, 0][:2]).astype("int")
    else:
        return None

# Loop utama
while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi citra ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dengan Cascade Classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Jika wajah ditemukan, lakukan eye tracking
    if len(faces) > 0:
        # Ambil ROI untuk mata kanan dan kiri
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

            for (ex, ey, ew, eh) in eyes:
                # Ambil ROI untuk mata kiri dan kanan
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Deteksi titik pupil
                pupil_coords = detect_pupil(eye_color)

                # Jika titik pupil ditemukan, gambarkan kotak pada mata dan lingkaran pada titik pupil
                if pupil_coords is not None:
                    (pupil_x, pupil_y) = pupil_coords
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.circle(roi_color, (ex + pupil_x, ey + pupil_y), 5, (0, 0, 255), -1)

    # Tampilkan citra dengan kotak mata dan lingkaran pupil
    cv2.imshow('Eye Tracking', frame)

    # Keluar dari program dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan semua objek
cap.release()
cv2.destroyAllWindows()