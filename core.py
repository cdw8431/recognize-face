import os
import cv2
from matplotlib import pyplot as plt

dir_path = os.path.abspath(os.path.dirname(__file__))

def get_classifier():
    data = os.path.join(dir_path, 'data', 'haarcascade_frontalface_default.xml')
    # data = os.path.join(dir_path, 'data', 'haarcascade_frontalface_alt.xml')
    # data = os.path.join(dir_path, 'data', 'haarcascade_eye_tree_eyeglasses.xml')
    # data = os.path.join(dir_path, 'data', 'haarcascade_profileface.xml')
    return cv2.CascadeClassifier(data)

def convert_image_color(image, color):
    return cv2.cvtColor(image, color)

def recognize_face(image):
    gray_image = convert_image_color(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=4)

def draw_reactangle(image, faces):
    if not len(faces): return
    color = (0,255,255)
    for(x, y, w, h) in faces:
        rect_start_position = (x,y)
        rect_end_position = (x+w, y+h)
        cv2.rectangle(image, rect_start_position, rect_end_position, color, thickness=2)

def show_face(image, faces):
    draw_reactangle(image, faces)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

def found_show_face(image):
    faces = recognize_face(image)
    show_face(image, faces)

def get_image(image):
    return cv2.imread(os.path.join(dir_path, "image", image))

if __name__ == "__main__":
    face_cascade = get_classifier()
    found_show_face(get_image("test5.jpg"))