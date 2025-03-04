import os
import cv2

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