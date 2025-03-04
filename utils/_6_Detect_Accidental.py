import cv2
import numpy as np
import os

def Accidental_Matching(Object_Pos_List, Base_Image, project_input, file_input, gs_num):
    """
    Matches accidental templates (sharp, flat, natural) to the detected objects in the musical score.

    :param Object_Pos_List: List of (x, y, w, h) bounding boxes of detected objects.
    :param Base_Image: The main image of the musical score.
    :param project_input: Project directory where results are stored.
    :param file_input: Folder name for accidental templates.
    :param gs_num: Identifier for saving the output image.
    :return: Updated Object_Pos_List, Accidental_Dict, Accidental_Detection_Image
    """
    Accidental_Template_Folder = os.path.join("Template", file_input, "Accidental_Template")
    Save_Folder = os.path.join(project_input, "Accidental_Matching")

    # Load accidental templates (sharp, flat, natural)
    template_files = [f for f in os.listdir(Accidental_Template_Folder) if f.endswith((".png", ".jpg"))]
    accidental_templates = {
        f: cv2.imread(os.path.join(Accidental_Template_Folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files
    }

    if not accidental_templates:
        print("No accidental templates found!")
        return Object_Pos_List, {}, Base_Image

    Accidental_Dict = {}
    detected_positions = []
    threshold = 0.7  # Similarity threshold
    accidental_detected_image = Base_Image.copy()

    # Iterate through detected object positions
    for (x, y, w, h) in Object_Pos_List:
        roi = Base_Image[y:y + h, x:x + w]  # Crop region of interest

        best_match = None
        best_template_name = None
        best_loc = None

        for template_name, template in accidental_templates.items():
            temp_h, temp_w = template.shape[:2]

            # Resize template to fit the ROI
            scale_w = w / temp_w
            scale_h = h / temp_h
            scale = min(scale_w, scale_h)  # Keep aspect ratio

            new_width = max(1, int(temp_w * scale))
            new_height = max(1, int(temp_h * scale))
            resized_template = cv2.resize(template, (new_width, new_height))

            # Apply Template Matching
            res = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Check if similarity is above threshold
            if max_val >= threshold and (best_match is None or max_val > best_match):
                best_match = max_val
                best_template_name = template_name
                best_loc = max_loc

        # If an accidental is detected, update dictionary and remove from object list
        if best_match is not None:
            detected_positions.append((x, y, w, h))  # Store detected positions
            center_x = x + w // 2
            center_y = y + h // 2
            accidental_type = best_template_name.split('.')[0]  # Extract accidental name from filename

            # Store in Accidental Dictionary
            Accidental_Dict[(center_x, center_y)] = {"Accidental_Type": accidental_type}

            # Draw bounding box and center point on detected accidental
            cv2.rectangle(accidental_detected_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
            cv2.circle(accidental_detected_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
            cv2.putText(accidental_detected_image, accidental_type, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Remove detected accidentals from Object_Pos_List
    Object_Pos_List = [pos for pos in Object_Pos_List if pos not in detected_positions]

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"Accidental_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, accidental_detected_image)

    print(f"Accidental detection image saved to: {save_path}")
    return Object_Pos_List, Accidental_Dict, accidental_detected_image
