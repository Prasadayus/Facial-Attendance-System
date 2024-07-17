#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import matplotlib.pyplot as plt

# Create a directory for images if it doesn't exist
os.makedirs("imgs", exist_ok=True)

cap = cv2.VideoCapture(0)
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img_index = 0

# Take client's name from the user
name = input("Enter client name: ")

# Validate the client's name input
name = "".join(c for c in name if c.isalnum() or c in [' ', '.', '-'])

# Find the latest img_index for the current client
for file_name in os.listdir("imgs/"):
    parts = file_name.split(".")
    if len(parts) == 3 and parts[0] == name:
        img_index = max(img_index, int(parts[1]))

if img_index == 0:
    img_index += 1

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to capture image")
        break

    key = cv2.waitKey(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceClassifier.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_cropped = img[y:y + h, x:x + w]
        img_index += 1
        face = cv2.resize(face_cropped, (450, 450))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        path = os.path.join("imgs", f"{name}.{int(img_index)}.jpg")
        cv2.imwrite(path, face)
        cv2.putText(img, str(img_index), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    2, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the output using matplotlib.pyplot
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('img')
    plt.show()

    if key == 113 or img_index >= 40:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




