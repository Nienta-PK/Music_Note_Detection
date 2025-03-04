"""
* Detect_Clef
* Assign the pitch of staff line according to the clef type
* Classify the Note Pitch

input:
* Object_Pos_List
* Base Image
* Clef_Template
* Upper/Lower staff Position

Output:
* Dict-Clef: clef-type, clef-position
* New_Object_Pos_List
* Clef-Annotated
* Assign the Upper_Staff
* Assign the Lower_Staff
"""

import cv2
import numpy as np
import os

def Clef_Matching(Object_Pos_List, Base_Image, project_input, file_input, gs_num):
    """
    Matches clef templates to the detected objects in the musical score.

    :param Object_Pos_List: List of (x, y, w, h) bounding boxes of detected objects.
    :param Base_Image: The main image of the musical score.
    :param Clef_Template_Folder: Folder containing clef templates.
    :param Save_Folder: Folder to save the detection result.
    :return: Updated Object_Pos_List, Clef_Dictionary, Clef_Detection_Image
    """
    Clef_Template_Folder = os.path.join("Template",file_input,"Clef_Template")
    Save_Folder = os.path.join(project_input,"Clef_Matching")

    # Load clef templates
    template_files = [f for f in os.listdir(Clef_Template_Folder) if f.endswith((".png", ".jpg"))]
    clef_templates = {f: cv2.imread(os.path.join(Clef_Template_Folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files}

    if not clef_templates:
        print("No clef templates found!")
        return Object_Pos_List, {}, Base_Image

    Clef_Dictionary = {}
    detected_positions = []
    threshold = 0.7  # Matching similarity threshold
    clef_detected_image = Base_Image.copy()

    # Iterate through detected object positions
    for (x, y, w, h) in Object_Pos_List:
        roi = Base_Image[y:y+h, x:x+w]  # Crop region of interest

        best_match = None
        best_template_name = None
        best_loc = None

        for template_name, template in clef_templates.items():
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

        # If a clef is detected, update dictionary and remove from object list
        if best_match is not None:
            detected_positions.append((x, y, w, h))  # Store detected positions
            Clef_Dictionary[(x, y, w,h)] = {"clef_type": best_template_name.split('.')[0]}

            # Draw bounding box around detected clef
            cv2.rectangle(clef_detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(clef_detected_image, best_template_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Remove detected clefs from Object_Pos_List
    Object_Pos_List = [pos for pos in Object_Pos_List if pos not in detected_positions]

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"Clef_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, clef_detected_image)

    print(f"Clef detection image saved to: {save_path}")
    return Object_Pos_List, Clef_Dictionary, clef_detected_image

def Staff_Pitch(Clef_Dictionary, Upper_Staff_Pos, Lower_Staff_Pos,gs_image):
    """
    Determines the pitch reference based on the detected clefs and staff positions.

    :param Clef_Dictionary: Dictionary with clef positions and types.
    :param Upper_Staff_Pos: Position of the upper staff.
    :param Lower_Staff_Pos: Position of the lower staff.
    :return: Pitch_Indicator_Dictionary
    """

    # Ensure the inputs are lists of 5 staff lines
    if len(Upper_Staff_Pos) != 5 or len(Lower_Staff_Pos) != 5:
        raise ValueError("Upper_Staff_Pos and Lower_Staff_Pos must each contain exactly 5 values.")

    Pitch_Indicator_Dictionary = {"Upper": {"Middle_Position": None, "Pitch": None},
                                  "Lower": {"Middle_Position": None, "Pitch": None}}
    global_middle_y = gs_image.shape[0] // 2
    for (x, y,w,h), clef_info in Clef_Dictionary.items():
        clef_type = clef_info["clef_type"]
        # Check if clef belongs to upper or lower staff
        if y < global_middle_y:  # Upper Staff
            staff_type = "Upper"
            staff_pos = Upper_Staff_Pos
        else:  # Lower Staff
            staff_type = "Lower"
            staff_pos = Lower_Staff_Pos

        # Assign pitch reference based on clef type
        if "sol" in clef_type.lower():  # Treble Clef (G-clef)
            pitch = "G"
            middle_y = staff_pos[3][1]+staff_pos[3][3]//2
        elif "fa" in clef_type.lower():  # Bass Clef (F-clef)
            pitch = "F"
            middle_y = staff_pos[1][1]+staff_pos[1][3]//2
        else:
            pitch = "Unknown"
            middle_y = 0

        # Store in dictionary
        Pitch_Indicator_Dictionary[staff_type]["Middle_Position"] = middle_y
        Pitch_Indicator_Dictionary[staff_type]["Pitch"] = pitch

        # Draw a dot on the middle_y position
        if middle_y > 0:
            cv2.circle(gs_image, (x + w // 2, middle_y), 5, (0, 0, 255), -1)  # Red dot

    # Save the visualization image
    cv2.imwrite("Dot_Indi.png", gs_image)

    return Pitch_Indicator_Dictionary