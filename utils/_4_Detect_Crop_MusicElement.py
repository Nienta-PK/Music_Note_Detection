"""
Input:
* Final Cleaned Image

Output
* Object_Position_List
"""
import os
import cv2
import numpy as np

def Music_Element_Detection(gs_image, binary_image, save_path, gs_num):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours: Remove small noise based on contour area
    min_contour_area = 400  # Adjust this value based on the size of noise
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create a blank mask to draw filtered contours
    filtered_mask = np.zeros_like(binary_image)
    cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Extract cleaned image
    cleaned_image = cv2.bitwise_and(binary_image, filtered_mask)

    # Convert grayscale to BGR for visualization
    final_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)

    # Draw bounding rectangles around remaining musical elements
    object_pos_list = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        object_pos_list.append((x, y, w, h))
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Green rectangles

    # Save the cleaned image
    os.makedirs(save_path, exist_ok=True)
    final_save_path = os.path.join(save_path, f"Detected_{gs_num+1}.png")
    cv2.imwrite(final_save_path, final_image)
    print(f"Final cleaned image saved to {final_save_path}")
    return object_pos_list,final_image

def Note_Head_Detection(gs_image, gs_num, object_pos_list, final_image, template_folder, file_input):
    """
    Detects note heads inside detected musical elements using template matching.

    :param gs_image: Original grayscale image.
    :param gs_num: Grayscale image index.
    :param object_pos_list: List of detected musical elements (bounding boxes).
    :param final_image: Image with bounding boxes for visualization.
    :param template_folder: Path to note head templates.
    :param file_input: The input file (for saving purposes).
    """
    # Load note head templates
    template_files = [f for f in os.listdir(template_folder) if f.endswith((".png", ".jpg"))]
    templates = {f: cv2.imread(os.path.join(template_folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files}

    # Ensure templates are loaded
    if not templates:
        print("No templates found in the folder!")
        return

    threshold = 0.6  # Minimum similarity score
    detected_rectangles = []  # Store detected bounding boxes

    for (x, y, w, h) in object_pos_list:
        # Extract the element region (ROI)
        roi = gs_image[y:y+h, x:x+w]

        # Apply template matching for each template
        for template_name, template in templates.items():
            temp_h, temp_w = template.shape[:2]

            # Resize template if it's wider than ROI
            if temp_w > roi.shape[1]:
                new_width = int(roi.shape[1])
                new_height = int(temp_h)
                template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                # Define the bounding box of the detected note head
                rect_x1 = x + pt[0]
                rect_y1 = y + pt[1]
                rect_x2 = rect_x1 + template.shape[1]
                rect_y2 = rect_y1 + template.shape[0]

                # Compute the center point of the detected note head
                center_x = (rect_x1 + rect_x2) // 2
                center_y = (rect_y1 + rect_y2) // 2

                # Skip this template if the center point falls inside any previously detected rectangle
                skip_template = False
                for (dx1, dy1, dx2, dy2) in detected_rectangles:
                    if dx1 <= center_x <= dx2 and dy1 <= center_y <= dy2:
                        skip_template = True
                        break

                if skip_template:
                    continue  # Skip duplicate detections

                # Store this detection
                detected_rectangles.append((rect_x1, rect_y1, rect_x2, rect_y2))

                # Draw blue rectangle around detected note head
                cv2.rectangle(final_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)  # Blue

                # Draw a small red circle at the center of the detected note head
                cv2.circle(final_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot

                # Label detected note head with template name
                cv2.putText(final_image, template_name, (rect_x1, rect_y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save the updated image
    save_path = os.path.join(f"{file_input}", "Detected_Note_Heads")
    os.makedirs(save_path, exist_ok=True)
    note_head_save_path = os.path.join(save_path, f"Note_Heads_Detected_{gs_num+1}.png")
    cv2.imwrite(note_head_save_path, final_image)
    print(f"Note heads detected and saved to {note_head_save_path}")



def Best_Matching(gs_image, object_pos_list, final_image, template_folder, file_input):
    """
    Detects note heads inside detected musical elements using template matching.
    
    :param gs_image: Original grayscale image
    :param object_pos_list: List of detected musical elements (bounding boxes)
    :param final_image: Image with bounding boxes for visualization
    :param template_folder: Path to note head templates
    :param file_input: The input file (for display purposes)
    """
    # Load note head templates
    template_files = [f for f in os.listdir(template_folder) if f.endswith((".png", ".jpg"))]
    templates = {f: cv2.imread(os.path.join(template_folder, f), cv2.IMREAD_GRAYSCALE) for f in template_files}

    # Ensure templates are loaded
    if not templates:
        print("No templates found in the folder!")
        return

    for (x, y, w, h) in object_pos_list:
        # Extract the element region (ROI)
        roi = gs_image[y:y+h, x:x+w]
        best_match = None  # Stores the best match details
        
        # Apply template matching for each template
        for template_name, template in templates.items():
            temp_h, temp_w = template.shape[:2]
            
            # Resize template if it's wider than ROI
            if temp_w > roi.shape[1]:
                new_width = int(roi.shape[1])
                new_height = int(temp_h)
                template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Check if this is the best match above the threshold
            threshold = 0.6  # Adjust sensitivity
            if max_val >= threshold and (best_match is None or max_val > best_match["score"]):
                best_match = {
                    "score": max_val,
                    "position": max_loc,
                    "size": template.shape,
                    "template_name": template_name
                }
        
        # If a valid best match is found, draw it
        if best_match:
            pt = best_match["position"]
            template_h, template_w = best_match["size"]
            template_name = best_match["template_name"]
            
            # Draw blue rectangle around detected note head
            cv2.rectangle(final_image, (x + pt[0], y + pt[1]), 
                          (x + pt[0] + template_w, y + pt[1] + template_h), 
                          (255, 0, 0), 2)  # Blue
            
            # Label detected note head with template name
            cv2.putText(final_image, template_name, (x + pt[0], y + pt[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save the updated image
    save_path = os.path.join(f"{file_input}", "Detected_Note_Heads")
    os.makedirs(save_path, exist_ok=True)
    note_head_save_path = os.path.join(save_path, "Note_Heads_Detected.png")
    cv2.imwrite(note_head_save_path, final_image)
    print(f"Note heads detected and saved to {note_head_save_path}")

def Template_Capture(gs_image,object_pos_list):
    # Convert grayscale to BGR for visualization
    final_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
    # Crop then save the image
    for idx, (x, y, w, h) in enumerate(object_pos_list):
        file_num = len(os.listdir("Template/Unlabeled_Template"))
        save_path = f"Template/Unlabeled_Template/template{file_num+1}.png"
        cropped_element = final_image[y:y+h, x:x+w]
        
        cv2.imwrite(save_path, cropped_element)
    
    print(f"Extracted {len(object_pos_list)} templates saved in {save_path}")

def main(file_input, gs_path):
    cleaned_image_path = os.path.join(file_input, "Only_Music_Elements")
    save_path = os.path.join(file_input, "Detected_Music_Elements")
    template_folder = os.path.join("Template", "Note_Head_Template")

    cleaned_images = [c for c in os.listdir(cleaned_image_path) if c.lower().endswith(".png")]
    grand_staffs = [g for g in os.listdir(gs_path) if g.lower().endswith(".png")]

    for gs_num, (grand_staff, binary_file) in enumerate(zip(grand_staffs, cleaned_images)):
        gs_image_path = os.path.join(gs_path, grand_staff)
        binary_image_path = os.path.join(cleaned_image_path, binary_file)

        gs_image = cv2.imread(gs_image_path, cv2.IMREAD_GRAYSCALE)
        binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

        if gs_image is None or binary_image is None:
            print(f"Error loading {grand_staff} or {binary_file}")
            continue

        # Process and save detected elements
        object_pos_list,annotated_image = Music_Element_Detection(gs_image, binary_image, save_path, gs_num)

        # Detect note heads inside elements
        Note_Head_Detection(gs_image, gs_num, object_pos_list, annotated_image, template_folder,file_input)

        # Crop and Save Template
        #Template_Capture(gs_image,object_pos_list)


if __name__ == "__main__":
    file_input = ["Twinkle_Twinkle_Little_Star","Happy_birth_day"]
    gs_path = ["Twinkle_Twinkle_Little_Star\Grand_Staff_IMG","Happy_birth_day\Grand_Staff_IMG"]
    x = 0
    main(file_input[x], gs_path[x])