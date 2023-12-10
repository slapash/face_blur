import cv2

def blur_face(image, factor=3.0):
    """
    Apply a blurring effect to the detected face in the image.
    The factor determines the intensity of the blur.
    """
    (h, w) = image.shape[:2]
    kW = int(w / factor) if int(w / factor) % 2 == 1 else int(w / factor) - 1
    kH = int(h / factor) if int(h / factor) % 2 == 1 else int(h / factor) - 1
    return cv2.GaussianBlur(image, (kW, kH), 0)

def detect_faces(image):
    """
    Detects faces in the given image using Haar Cascades.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def blur_faces_in_image(image, faces):
    """
    Apply blur to each detected face in the image.
    """
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        image[y:y+h, x:x+w] = blur_face(face_region, factor=3.0)
    return image

def manual_blur(image, x, y, w, h, factor=3.0):
    """
    Manually apply blur to a specified region in the image.
    """
    region = image[y:y+h, x:x+w]
    blurred_region = blur_face(region, factor)
    image[y:y+h, x:x+w] = blurred_region
    return image
