import cv2
import numpy as np
import face_recognition
import os
import threading
import websocket
import time

# ——————————————————————————————————————————————————————————————
# 1) تحميل وصنع encodings للوجوه من مجلّد 'persons'
path = 'persons'
images = []
classNames = []

for fname in os.listdir(path):
    img = cv2.imread(os.path.join(path, fname))
    if img is None:
        continue
    images.append(img)
    classNames.append(os.path.splitext(fname)[0])

def findEncodings(imgs):
    encs = []
    for im in imgs:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_encodings(rgb)
        if boxes:
            encs.append(boxes[0])
    return encs

knownEncodings = findEncodings(images)
print("✅ Encoding Complete for:", classNames)

# ——————————————————————————————————————————————————————————————
# 2) فتح WebSocket لاستقبال فريمات JPEG
latest_frame = None

def on_message(ws, message):
    global latest_frame
    # الرسالة هنا بايناري JPEG
    arr = np.frombuffer(message, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    latest_frame = img

def on_open(ws):
    print("🔌 WebSocket opened to /Camera")

def on_error(ws, error):
    print("❌ WebSocket error:", error)

def on_close(ws, code, reason):
    print("📴 WebSocket closed")

def start_ws():
    ws_url = "ws://192.168.4.1/Camera"
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

# شغّل WebSocket في ثريد daemon
threading.Thread(target=start_ws, daemon=True).start()
time.sleep(2)  # دقائق للاتصال

# ——————————————————————————————————————————————————————————————
# 3) اللوب الرئيسي: عرض الفيديو مع التعرف على الوجوه
print("▶️ Starting main loop. Press 'q' to quit.")
while True:
    if latest_frame is None:
        time.sleep(0.01)
        continue

    frame = latest_frame.copy()
    # خفّض دقّة الفريم للتسريع
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # اكتشاف مواقع الوجوه و encodings
    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    for enc, loc in zip(face_encs, face_locs):
        distances = face_recognition.face_distance(knownEncodings, enc)
        idx = np.argmin(distances)
        if distances[idx] < 0.6:
            name = classNames[idx].upper()
            y1, x2, y2, x1 = [v * 4 for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('ESP32-CAM Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
