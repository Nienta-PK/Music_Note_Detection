import cv2
import os


def Room_Matching(Object_Pos_List, Base_Image, project_input, file_input, gs_num):
    """
    Matches clef templates to the detected objects in the musical score.

    :param Object_Pos_List: List of (x, y, w, h) bounding boxes of detected objects.
    :param Base_Image: The main image of the musical score.
    :param Clef_Template_Folder: Folder containing clef templates.
    :param Save_Folder: Folder to save the detection result.
    :return: Updated Object_Pos_List, Clef_Dictionary, Clef_Detection_Image
    """
    Clef_Template_Folder = os.path.join("Template",file_input,"Room_Template")
    Save_Folder = os.path.join(project_input,"Room_Matching")

    # Load clef templates
    template_files = [f for f in os.listdir(Clef_Template_Folder) if f.endswith((".png", ".jpg"))]
    clef_templates = {f: cv2.imread(os.path.join(Clef_Template_Folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files}

    if not clef_templates:
        print("No Room templates found!")
        return Object_Pos_List, {}, Base_Image

    Room_Type = "None"
    detected_positions = []
    threshold = 0.7  # Matching similarity threshold
    room_detected_image = Base_Image.copy()

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
            Room_Type = best_template_name.split('.')[0]

            # Draw bounding box around detected clef
            cv2.rectangle(room_detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(room_detected_image, best_template_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Remove detected clefs from Object_Pos_List
    Object_Pos_List = [pos for pos in Object_Pos_List if pos not in detected_positions]

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"Room_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, room_detected_image)

    print(f"Room detection image saved to: {save_path}")
    return Object_Pos_List, Room_Type, room_detected_image