import pyautogui
import cv2
import pytesseract
import numpy as np
import subprocess

COULEUR_ROUGE = (219, 58, 52)
COULEUR_JAUNE = (247, 183, 53)
COULEUR_BLEUE = (8, 76, 97)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\quent\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def write_autoit_script(autoit_code, script_path):
    with open(script_path, "w") as file:
        file.write(autoit_code)

def run_autoit_script(script_path):
    subprocess.Popen([r'C:\Program Files (x86)\AutoIt3\AutoIt3.exe', script_path])

def get_grid_region():
    mouse_pos = []

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos.append(x)
            mouse_pos.append(y)

    screenshot = pyautogui.screenshot()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow("image", np.array(screenshot))
    cv2.setMouseCallback("image", on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return tuple(mouse_pos)


def to_gray(screenshot):
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return threshold

def crop_the_grid(screenshot, keep_gray:bool=True):
    threshold = to_gray(screenshot)

    # Crop screen to get grid only
    start_row = 0
    end_row = threshold.shape[0] - 1
    start_col = 0
    end_col = threshold.shape[1] - 1

    for i, row in enumerate(threshold):
        if len(np.nonzero(row == 255)[0]) > 1:
            start_row = i
            break

    for i, row in enumerate(threshold[::-1]):
        if len(np.nonzero(row == 255)[0]) > 1:
            end_row = threshold.shape[0] - i - 1
            break

    for i, col in enumerate(threshold.T):
        if len(np.nonzero(col == 255)[0]) > 1:
            start_col = i
            break

    for i, col in enumerate(threshold[:, ::-1].T):
        if len(np.nonzero(col == 255)[0]) > 1:
            end_col = threshold.shape[1] - i - 1
            break

    if keep_gray:
        threshold = threshold[start_row:end_row, start_col:end_col]
        return threshold
    else:
        screenshot = screenshot[start_row:end_row, start_col:end_col, :]
        return screenshot
    
def take_screenshot(region, grid_crop:bool=False, keep_gray:bool=False):    
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot.crop(region))
    if grid_crop:
        screenshot = crop_the_grid(screenshot, keep_gray=keep_gray)
    return screenshot

def remove_grid(screenshot, threshold=0.7):
    # Get the indexes of the white columns
    white_cols_idx = np.where(screenshot.sum(axis=0) >= 255*screenshot.shape[0]*threshold)[0]
    white_rows_idx = np.where(screenshot.sum(axis=1) >= 255*screenshot.shape[0]*threshold)[0]

    # Remove the white rows/columns
    screenshot[:, white_cols_idx] = 0
    screenshot[white_rows_idx, :] = 0

    return screenshot

def get_tentative(screenshot, tentative_num):
    L = screenshot.shape[0]//6
    return screenshot[L*tentative_num:L*(tentative_num+1), :]


def get_text_from_screen(screenshot):
    text = pytesseract.image_to_string(screenshot, config='--psm 6', lang="fra")
    text = text.replace("\n", "").replace(" ", "")
    return text

def get_text_from_tentative(region, tentative_num):
    screenshot = take_screenshot(region, grid_crop=True, keep_gray=True)
    screenshot = remove_grid(screenshot)
    screenshot = get_tentative(screenshot, tentative_num)
    return get_text_from_screen(screenshot)

def get_dominant_color(image):
    # Reshape the image array to 2D
    image_2d = image.reshape(-1, 3)

    # Find unique colors and their counts
    colors, counts = np.unique(image_2d, axis=0, return_counts=True)

    # Find the index of the dominant color
    dominant_color_index = np.argmax(counts)

    # Retrieve the dominant color
    dominant_color = colors[dominant_color_index]

    return tuple(dominant_color)

def get_letter_result(image_letter):
    dominant_color = get_dominant_color(image_letter)
    if dominant_color == COULEUR_ROUGE:
        return 1
    elif dominant_color == COULEUR_JAUNE:
        return 0
    elif dominant_color == COULEUR_BLEUE:
        return -1
    else:
        raise Exception("Couleur non reconnue")
    
def get_word_result(image_word, length):
    L = image_word.shape[1]//length
    return [get_letter_result(image_word[:, i*L:(i+1)*L]) for i in range(length)]
    