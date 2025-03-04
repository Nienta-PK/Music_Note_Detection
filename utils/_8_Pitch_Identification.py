import cv2
import numpy as np
import os

def NoteHead_Matching(Beat_Dictionary, Base_Image, project_input, file_input, gs_num):
    """
    Matches notehead templates within the bounding boxes defined in Beat_Dictionary.
    Allows multiple detections per region and prevents duplicate detections.

    :param Beat_Dictionary: Dictionary of detected beats with their bounding boxes.
    :param Base_Image: The main image of the musical score.
    :param project_input: Project directory where results are stored.
    :param file_input: Folder name for notehead templates.
    :param gs_num: Identifier for saving the output image.
    :return: NoteHead_Dictionary, NoteHead_Detection_Image
    """
    NoteHead_Template_Folder = os.path.join("Template", file_input, "Note_Head_Template")
    Save_Folder = os.path.join(project_input, "Note_Head_Matching")

    # Load notehead templates (whole_note, half_note, quarter_note, etc.)
    template_files = [f for f in os.listdir(NoteHead_Template_Folder) if f.endswith((".png", ".jpg"))]
    notehead_templates = {
        f: cv2.imread(os.path.join(NoteHead_Template_Folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files
    }

    if not notehead_templates:
        print("No notehead templates found!")
        return {}, Base_Image

    NoteHead_Dictionary = {}  # Store detected noteheads
    detected_rectangles = []  # To store bounding boxes for duplicate prevention
    threshold = 0.6  # Matching similarity threshold

    # Ensure Base_Image is in BGR format for visualization
    if len(Base_Image.shape) == 2:  # If grayscale
        notehead_detected_image = cv2.cvtColor(Base_Image, cv2.COLOR_GRAY2BGR)
    else:
        notehead_detected_image = Base_Image.copy()  # Already in BGR

    # Iterate through beat regions in Beat_Dictionary.keys()
    for (x, y, w, h) in Beat_Dictionary.keys():
        roi = Base_Image[y:y + h, x:x + w]  # Crop region of interest

        # Ensure ROI is grayscale
        if len(roi.shape) == 3:  
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        for template_name, template in notehead_templates.items():
            temp_h, temp_w = template.shape[:2]

            # Resize template to fit the ROI
            scale = min(w / temp_w, h / temp_h)  # Keep aspect ratio
            new_width = max(1, int(temp_w * scale))
            new_height = max(1, int(temp_h * scale))
            resized_template = cv2.resize(template, (new_width, new_height))

            # Ensure resized template is uint8 for matchTemplate()
            resized_template = resized_template.astype(np.uint8)

            # Apply Template Matching
            res = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                # Compute absolute position of detected notehead
                detected_x = x + pt[0]
                detected_y = y + pt[1]
                detected_w, detected_h = new_width, new_height
                center_x = detected_x + detected_w // 2
                center_y = detected_y + detected_h // 2

                # Allow multiple detections but prevent overlapping duplicates
                duplicate = any(dx1 <= center_x <= dx2 and dy1 <= center_y <= dy2 
                                for dx1, dy1, dx2, dy2 in detected_rectangles)
                if duplicate:
                    continue  # Skip this detection

                # Store the detection
                detected_rectangles.append((detected_x, detected_y, detected_x + detected_w, detected_y + detected_h))
                NoteHead_Dictionary[(detected_x, detected_y, detected_w, detected_h)] = {
                    "Note_Type": template_name.split('.')[0], 
                    "Center_Point": (center_x, center_y)
                }

                # Draw bounding box exactly around detected notehead
                cv2.rectangle(notehead_detected_image, (detected_x, detected_y), 
                              (detected_x + detected_w, detected_y + detected_h), (0, 255, 0), 2)  # Green box
                cv2.circle(notehead_detected_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
                cv2.putText(notehead_detected_image, template_name.split('.')[0], 
                            (detected_x, detected_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"NoteHead_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, notehead_detected_image)

    print(f"NoteHead detection image saved to: {save_path}")
    return NoteHead_Dictionary, notehead_detected_image

def Pitch_Detection():
    pass
