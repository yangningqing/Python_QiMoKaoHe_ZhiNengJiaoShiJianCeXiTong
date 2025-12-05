
from typing import Optional

import cv2
from pyzbar import pyzbar  # type: ignore


def _decode_bytes(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip()


def decode_qr_from_camera(timeout_seconds: int = 8) -> Optional[str]:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    start = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    result: Optional[str] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)
        if decoded_objects:
            raw = decoded_objects[0].data
            text = _decode_bytes(raw)
            result = text
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("QR Code Sign-In", frame)
            cv2.waitKey(300)
            break

        cv2.imshow("QR Code Sign-In (按 q 退出)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = (cv2.getTickCount() - start) / freq
        if elapsed > timeout_seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    return result

