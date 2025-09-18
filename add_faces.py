import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

name = input("Enter your name: ").strip().lower()
folder = f"embeddings/{name}"
os.makedirs(folder, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

cam = cv2.VideoCapture(0)
count = 0
print("[INFO] Starting FaceNet-PyTorch embedding capture. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (160, 160))
            face_tensor = torch.tensor(face_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            face_tensor = face_tensor.to(device)
            embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            count += 1
            file_path = f"{folder}/{name}_{count}.npy"
            np.save(file_path, embedding)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("FaceNet-PyTorch Embedding Capture", frame)
    if cv2.waitKey(1) == ord('q') or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved {count} embeddings to {folder}")
