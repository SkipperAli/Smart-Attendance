import cv2
import os
from deepface import DeepFace

students_path = "students"

print("Starting camera...")
cap = cv2.VideoCapture(0)

match_found = False 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Smart Attendance", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        print("Scanning...")

        for student in os.listdir(students_path):
            student_path = os.path.join(students_path, student)

            try:
                result = DeepFace.verify(
                    frame,
                    student_path,
                    enforce_detection=False
                )

                if result["verified"]:
                    name = os.path.splitext(student)[0]
                    print(f"WELCOME BACK {name}")
                    match_found = True
                    break  

            except Exception as e:
                print(f"Error processing {student}: {e}")
                continue

        if not match_found:
            print("WHO ARE YOU IMPOSTER!!!!!!")

        break  

cap.release()
cv2.destroyAllWindows()
