import cv2
import easyocr
import numpy as np
import re
import unicodedata

# Global variable to count the number of dots detected
dot_count = 0

# Map ambiguous letters to digits
confusion_map = {
    'S': '5', 'B': '8', 'O': '0', 'I': '1', 'Z': '2', 'G': '6'
}

# Map dotted versions back to alphabet letters
dot_letter_map = {
    'Ạ': 'A', 'Ḅ': 'B', 'C̣': 'C', 'Ḍ': 'D', 'Ẹ': 'E', 'F̣': 'F', 'G̣': 'G',
    'Ḥ': 'H', 'Ị': 'I', 'J̣': 'J', 'Ḳ': 'K', 'Ḷ': 'L', 'Ṃ': 'M', 'Ṇ': 'N',
    'Ọ': 'O', 'P̣': 'P', 'Q̣': 'Q', 'Ṛ': 'R', 'Ṣ': 'S', 'Ṭ': 'T', 'Ụ': 'U',
    'Ṿ': 'V', 'Ẉ': 'W', 'X̣': 'X', 'Ỵ': 'Y', 'Ẓ': 'Z'
}

# For debugging: Save crops where dots were found
DEBUG = False

def detect_dot_below(image, box):
    global dot_count
    (top_left, top_right, bottom_right, bottom_left) = box
    bottom_y = int(max(bottom_left[1], bottom_right[1]))
    center_x = int((bottom_left[0] + bottom_right[0]) / 2)
    height = int(abs(bottom_y - top_left[1]))
    width = int(abs(top_right[0] - top_left[0]))

    # Define region just below the character
    margin = 3
    dot_region = image[
        bottom_y + margin : bottom_y + margin + height // 2,
        center_x - width // 4 : center_x + width // 4
    ]

    if dot_region.size == 0:
        return False

    # Process and look for circular shapes
    dot_gray = cv2.cvtColor(dot_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(dot_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3 < area < 30:  # dot-sized contour
            dot_count += 1  # Increment the global dot count
            if DEBUG:
                cv2.imwrite(f"dot_debug_{dot_count}.png", dot_region)
            return True
    return False

def process_character_with_box(char, image, box):
    if char in dot_letter_map:
        return dot_letter_map[char]
    elif char in confusion_map:
        if detect_dot_below(image, box):
            return char  # Keep as letter if dot detected below
        else:
            return confusion_map[char]  # Map to number if no dot detected
    else:
        return char

def normalize_text_with_boxes(text, image, box):
    result = ""
    for i, char in enumerate(text):
        result += process_character_with_box(char, image, box)
    return result

def extract_code_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(blurred, -1, sharpen_kernel)

    # Using EasyOCR for text detection
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(sharp)

    if not results:
        print("No text detected.")
        return None

    best_candidate = None
    best_score = 0
    all_detected_text = []
    pattern = re.compile(r'[\w\-]{4,}', re.UNICODE)

    # Loop through the detected text
    for bbox, text, prob in results:
        all_detected_text.append(text.strip())
        words = text.strip().split()
        for word in words:
            normalized = normalize_text_with_boxes(word, image, bbox)

            digits = sum(c.isdigit() for c in normalized)
            alphabets = sum(c.isalpha() for c in normalized)

            if pattern.fullmatch(normalized) and digits >= 3 and alphabets >= 1:
                score = prob * len(normalized)
                if score > best_score:
                    best_score = score
                    best_candidate = normalized

    if best_candidate:
        print("→ Final normalized output:", best_candidate)
    else:
        print("No valid code found. Detected text:")
        for line in all_detected_text:
            print("•", line)

    print(f"Total dots detected below characters: {dot_count}")
    return best_candidate

# Example usage
extract_code_from_image(r"D:\nokia obect detection\Screenshot 2025-04-16 140213.png")
