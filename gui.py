import PySimpleGUI as sg
import cv2
import os
import io
from PIL import Image
from image_processor import detect_faces, blur_faces_in_image, manual_blur

def convert_to_bytes(file_or_bytes, resize=None):
    """
    Convert the given file or bytes to bytes that can be displayed in the GUI.
    """
    if isinstance(file_or_bytes, str):
        img = Image.open(file_or_bytes)
    else:
        try:
            img = Image.open(io.BytesIO(file_or_bytes))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), Image.Resampling.LANCZOS)

    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()

# Define the window's contents (layout)
layout = [
    [sg.Text("Choose an image to blur faces:")],
    [sg.Input(), sg.FileBrowse(key="-FILE-")],
    [sg.Text("Select output folder:"), sg.Input(), sg.FolderBrowse(key="-FOLDER-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Button("Blur Faces"), sg.Button("Manual Edit"), sg.Button("Exit")]
]

# Create the window
window = sg.Window("Face Blur Application", layout, size=(600, 500))

# This variable will hold the image that is currently being processed
current_image = None

# Event loop
while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Blur Faces":
        file_path = values["-FILE-"]
        output_folder = values["-FOLDER-"]
        if file_path and output_folder:
            current_image = cv2.imread(file_path)
            faces = detect_faces(current_image)
            current_image = blur_faces_in_image(current_image, faces)

            img_bytes = cv2.imencode('.png', current_image)[1].tobytes()
            window["-IMAGE-"].update(data=convert_to_bytes(img_bytes, resize=(400, 400)))

            filename = os.path.basename(file_path)
            output_path = os.path.join(output_folder, f"blurred_{filename}")
            cv2.imwrite(output_path, current_image)

            sg.popup(f"Image saved to {output_path}")

    if event == "Manual Edit":
        if current_image is None:
            sg.popup_error('No image loaded. Please load an image first.')
            continue

        layout_manual = [
            [sg.Text('Enter the coordinates and size of the area to blur:')],
            [sg.Text('X:', size=(2, 1)), sg.InputText(key='X', size=(5, 1)), sg.Text('Y:', size=(2, 1)), sg.InputText(key='Y', size=(5, 1))],
            [sg.Text('Width:', size=(5, 1)), sg.InputText(key='Width', size=(5, 1)), sg.Text('Height:', size=(5, 1)), sg.InputText(key='Height', size=(5, 1))],
            [sg.Button('Apply'), sg.Button('Cancel')]
        ]

        window_manual = sg.Window('Manual Edit', layout_manual)

        while True:
            event_manual, values_manual = window_manual.read()

            if event_manual == sg.WIN_CLOSED or event_manual == 'Cancel':
                break
            if event_manual == 'Apply':
                try:
                    x = int(values_manual['X'])
                    y = int(values_manual['Y'])
                    width = int(values_manual['Width'])
                    height = int(values_manual['Height'])
                    current_image = manual_blur(current_image, x, y, width, height)

                    img_bytes = cv2.imencode('.png', current_image)[1].tobytes()
                    window["-IMAGE-"].update(data=convert_to_bytes(img_bytes, resize=(400, 400)))

                except ValueError:
                    sg.popup_error('Please enter valid integer values.')

        window_manual.close()

# Finish up by removing from the screen
window.close()
