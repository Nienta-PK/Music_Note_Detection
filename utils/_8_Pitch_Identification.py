import cv2
import numpy as np
import os
import math

def NoteHead_Matching(Beat_Dictionary, Base_Image, project_input, file_input, gs_num):
    """
    Matches notehead templates within the bounding boxes defined in Beat_Dictionary.
    Allows multiple detections per region, prevents duplicate detections, and adds Beat Type.

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
    threshold = 0.55  # Matching similarity threshold

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

        # Extract the Beat Type from Beat_Dictionary
        beat_type = Beat_Dictionary[(x, y, w, h)]['Beat_Type'].split('_')[0]

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

                # Store the detection with Beat Type
                detected_rectangles.append((detected_x, detected_y, detected_x + detected_w, detected_y + detected_h))
                NoteHead_Dictionary[(detected_x, detected_y, detected_w, detected_h)] = {
                    "Note_Type": template_name.split('.')[0], 
                    "Center_Point": (center_x, center_y),
                    "Beat": beat_type
                }

                # Draw bounding box exactly around detected notehead
                cv2.rectangle(notehead_detected_image, (detected_x, detected_y), 
                              (detected_x + detected_w, detected_y + detected_h), (0, 255, 0), 2)  # Green box
                cv2.circle(notehead_detected_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
                cv2.putText(notehead_detected_image, f"{template_name.split('.')[0]} ({beat_type})", 
                            (detected_x, detected_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save result image
    os.makedirs(Save_Folder, exist_ok=True)
    save_path = os.path.join(Save_Folder, f"NoteHead_Detection_{gs_num+1}.png")
    cv2.imwrite(save_path, notehead_detected_image)

    print(f"NoteHead detection image saved to: {save_path}")
    return NoteHead_Dictionary, notehead_detected_image


def Pitch_Detection(NoteHead_Dictionary, Base_Image, Pitch_Indicator_Dictionary, Staff_Spacing):
    """
    Detects pitch and octave shift based on notehead center position relative to staff position.

    :param NoteHead_Dictionary: Dictionary containing detected noteheads with beat and center position.
    :param Base_Image: The main image of the musical score.
    :param Pitch_Indicator_Dictionary: Stores staff mid positions and reference pitches.
    :param Staff_Spacing: Distance between staff lines.
    :return: Updated NoteHead_Dictionary with corrected pitch information and octave multipliers.
    """
    
    Pitch_List = ["C", "D", "E", "F", "G", "A", "B"]  # Standard musical pitch cycle
    
    # Iterate through detected noteheads
    for (x, y, w, h), note_info in NoteHead_Dictionary.items():
        note_cp_x, note_cp_y = note_info["Center_Point"]

        # Determine if the note belongs to the upper or lower staff
        if note_cp_y < Base_Image.shape[0] // 2:
            Stave = Pitch_Indicator_Dictionary["Upper"]
        else:
            Stave = Pitch_Indicator_Dictionary["Lower"]

        Stave_mid_pos = Stave["Middle_Position"]
        Stave_Pitch = Stave["Pitch"]

        # Select the correct pitch reference list
        if Stave_Pitch == "G":  # Treble Clef (G-clef)
            Pitch_List_Ref = ["G", "F", "E", "D", "C", "B", "A"]  # **Inverted sequence**
            start_pos = 4
        elif Stave_Pitch == "F":  # Bass Clef (F-clef)
            Pitch_List_Ref = ["F", "E", "D", "C", "B", "A", "G"]  # **Inverted sequence**
            start_pos = 3
        else:
            NoteHead_Dictionary[(x, y, w, h)]["Pitch"] = "Unknown"
            NoteHead_Dictionary[(x, y, w, h)]["Multiplier"] = 0
            continue

        # Calculate position shift in terms of staff spaces
        Space = abs(Staff_Spacing // 2)  # Half-space distance for note transitions
        y_difference = Stave_mid_pos - note_cp_y  # **Invert direction**
        time = round(y_difference / Space)  # Number of spaces away from the middle

        # Estimate reference position for error correction
        estimated_ref_pos = Stave_mid_pos - time * Space  # **Invert direction**
        error_margin = Staff_Spacing // 4  # Acceptable deviation from expected position

        # Determine the octave shift
        multiplier = (time + start_pos) // 7  # **Determine octave shift**

        # If the note is within the expected range, determine the pitch
        if estimated_ref_pos - error_margin <= note_cp_y <= estimated_ref_pos + error_margin:
            pitchList_pos = (time + start_pos) % 7
            pitch = Pitch_List[pitchList_pos]
        else:
            pitch = "Unknown"  # If outside expected error range

        # Store pitch and multiplier in the NoteHead_Dictionary
        NoteHead_Dictionary[(x, y, w, h)]["Pitch"] = pitch
        NoteHead_Dictionary[(x, y, w, h)]["Multiplier"] = multiplier

    return NoteHead_Dictionary

def Save_Pitch_Annotated(NoteHead_Dictionary, gs_num, gs_image, project_path):
    """
    Saves an annotated image with detected noteheads and their corresponding pitch labels.

    :param NoteHead_Dictionary: Dictionary containing detected noteheads with pitch information.
    :param gs_num: Identifier for saving the output image (e.g., page number).
    :param gs_image: The original grayscale sheet music image.
    :param output_folder: Folder where the annotated image will be saved.
    """

    # Convert grayscale image to BGR for annotation
    annotated_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h), note_info in NoteHead_Dictionary.items():
        center_x, center_y = note_info["Center_Point"]
        pitch = note_info.get("Pitch", "Unknown")  # Default to "Unknown" if pitch not found

        # Draw bounding box around detected notehead
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

        # Draw center point
        cv2.circle(annotated_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot

        # Label the detected pitch near the note
        cv2.putText(annotated_image, pitch, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # Ensure the output folder exists
    output_folder = os.path.join(project_path, "Pitch_Annotated")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the annotated image
    save_path = os.path.join(output_folder, f"Pitch_Annotated_{gs_num+1}.png")
    cv2.imwrite(save_path, annotated_image)
    
    print(f"Pitch annotated image saved to: {save_path}")
    
    return save_path  # Return the saved image path
