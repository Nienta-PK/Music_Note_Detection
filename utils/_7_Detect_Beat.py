import cv2
import numpy as np
import os

def Beat_Matching(Object_Pos_List, Base_Image, project_input, file_input, gs_num):
    """
    Matches beat period templates (4_beat, 2_beat, 1_beat, 1/2_beat, 1/4_beat, etc.) in the musical score.

    :param Object_Pos_List: List of (x, y, w, h) bounding boxes of detected objects.
    :param Base_Image: The main image of the musical score.
    :param project_input: Project directory where results are stored.
    :param file_input: Folder name for beat templates.
    :param gs_num: Identifier for saving the output image.
    :return: Updated Object_Pos_List, Beat_Dict, Beat_Detection_Image
    """
    Beat_Template_Folder = os.path.join("Template", file_input, "Beat_Template")
    Save_Folder = os.path.join(project_input, "Beat_Matching")

    # Load beat period templates
    template_files = [f for f in os.listdir(Beat_Template_Folder) if f.endswith((".png", ".jpg"))]
    beat_templates = {
        f: cv2.imread(os.path.join(Beat_Template_Folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files
    }

    if not beat_templates:
        print("No beat templates found!")
        return Object_Pos_List, {}, Base_Image

    Beat_Dict = {}  # Store detected beats
    detected_positions = []
    threshold = 0.4  # Matching similarity threshold
    print(Base_Image.shape)
    if len(Base_Image.shape) == 2:
        beat_detected_image = cv2.cvtColor(Base_Image, cv2.COLOR_GRAY2BGR)
    else:
        beat_detected_image = Base_Image.copy()  # Already in BGR

    # Iterate through detected object positions
    for (x, y, w, h) in Object_Pos_List:
        roi = Base_Image[y:y + h, x:x + w]  # Crop region of interest

        best_match = None
        best_template_name = None
        best_loc = None

        for template_name, template in beat_templates.items():
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

        # If a beat is detected, update dictionary and remove from object list
        if best_match is not None:
            detected_positions.append((x, y, w, h))  # Store detected positions
            beat_type = best_template_name.split('.')[0]  # Extract beat name from filename
            center_x = x + w // 2
            center_y = y + h // 2

            # Store in Beat Dictionary using bounding box as key
            Beat_Dict[(x, y, w, h)] = {"Beat_Type": beat_type, "Center_Point": (center_x, center_y)}

            # Draw bounding box, center point, and label on detected beat
            cv2.rectangle(beat_detected_image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box
            cv2.circle(beat_detected_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
            cv2.putText(beat_detected_image, beat_type, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Remove detected beats from Object_Pos_List
    Object_Pos_List = [pos for pos in Object_Pos_List if pos not in detected_positions]

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"Beat_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, beat_detected_image)

    print(f"Beat detection image saved to: {save_path}")
    return Object_Pos_List, Beat_Dict, beat_detected_image
