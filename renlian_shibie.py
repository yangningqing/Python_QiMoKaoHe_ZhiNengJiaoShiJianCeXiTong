#--------------------------------------------负责人：杨宁轻------------------------------------------------#
import os
import cv2
import numpy as np

FACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "face_data")
CASCADE_PATH = os.path.join(FACE_DATA_DIR, "haarcascade_frontalface_alt.xml")
TRAINER_PATH = os.path.join(FACE_DATA_DIR, "trainer.yml")
FACE_LIST_PATH = os.path.join(FACE_DATA_DIR, "face_list.txt")


def load_face_dictionary(list_path=FACE_LIST_PATH):
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"未找到人脸字典文件：{list_path}")
    data = np.loadtxt(list_path, dtype="str")
    mapping = {}
    for row in data:
        mapping[int(row[0])] = row[1]
    return mapping


def recognize_from_camera(duration_seconds=10, on_identity=None, silent=False):

    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"未找到 Haar 分类器：{CASCADE_PATH}")
    if not os.path.exists(TRAINER_PATH):
        raise FileNotFoundError(f"未找到训练模型：{TRAINER_PATH}")

    face_dict = load_face_dictionary()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    collected = set()
    last_identity = None
    start = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    if not silent:
        print("摄像头人脸识别已启动，按 'q' 退出窗口。")

    while True:
        success, frame = cap.read()
        if not success:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id_face, confidence = recognizer.predict(gray[y : y + h, x : x + w])
            if confidence < 100:
                identity = face_dict.get(id_face, "unknown")
            else:
                identity = "unknown"
            last_identity = identity
            if identity != "unknown":
                collected.add(identity)
            label = f"{identity} {confidence:.2f}"
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elapsed = (cv2.getTickCount() - start) / freq
        if not silent:
            cv2.imshow("Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if duration_seconds is not None and elapsed >= duration_seconds:
            break

    cap.release()
    if not silent:
        cv2.destroyAllWindows()

    if on_identity:
        on_identity(last_identity)
    return last_identity, collected


if __name__ == "__main__":
    recognize_from_camera()

