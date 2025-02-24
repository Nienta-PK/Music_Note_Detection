"""
1.Go into grand_staff_number floder and see through each grand_staff
2.Get Line Position using contour
3.Remove the horizontal and vertical line to get the image of only music element
4.Using Contour to detect all music element
3.Experiment...
4.Change the music element to list/tuple/dictionary data structure in sequence
5.
"""
import os 
import cv2
import numpy as np

def Line_Position(gs_folder,gs,gs_num):
    image_path = os.path.join(gs_folder, gs)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return [],[],None,None
    # Step 1: Binary thresholding
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        
    # Step 2: Detect horizontal lines (staff lines)
    kernel_width = max(image.shape[1] // 30, 1)  # ensure kernel width is at least 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    print(f"Grandstaff {gs_num+1} has extracted the lines successfully")

    # Step 3: Use contours to bound each line and separate them as Upper and Lower Staff
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return [],[],None,None
    
    # Get bounding boxes for each detected contour (line)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    # Sort the bounding boxes by their y-coordinate (top of the box)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    
    # Compute gaps between consecutive boxes (difference between the bottom of one and the top of the next)
    gaps = []
    for i in range(len(bounding_boxes) - 1):
        # bottom of current box = y + h
        gap = bounding_boxes[i+1][1] - (bounding_boxes[i][1] + bounding_boxes[i][3])
        gaps.append(gap)
    
    # Assume the largest gap is the separation between upper and lower staff.
    if gaps:
        max_gap = max(gaps)
        separation_index = gaps.index(max_gap) + 1
    else:
        separation_index = len(bounding_boxes) // 2  # fallback if gaps list is empty

    # Separate bounding boxes into upper and lower staves
    upper_staff_boxes = bounding_boxes[:separation_index]
    lower_staff_boxes = bounding_boxes[separation_index:]
    
    # Convert grayscale image to BGR for annotation
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw green rectangles for upper staff lines
    for box in upper_staff_boxes:
        x, y, w, h = box
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Draw red rectangles for lower staff lines
    for box in lower_staff_boxes:
        x, y, w, h = box
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Save the annotated image
    annotated_path = os.path.join(gs_folder, "Anotated_Grandstaff")
    os.makedirs(annotated_path,exist_ok=True)
    save_path =  os.path.join(gs_folder, "Anotated_Grandstaff",f"Annotated_Grandstaff_{gs_num+1}.png")
    cv2.imwrite(save_path, annotated_image)
    print(f"Annotated image saved to {save_path}")
    
    # Return the separated bounding boxes and the annotated image for further processing if needed.
    return upper_staff_boxes, lower_staff_boxes, annotated_path, detected_lines, binary

def Music_Element(gs_path,gs_num,detected_line,bw_grandstaff):
    
    # Step 1: Substract Horizontal line
    cleaned_horizontal = cv2.subtract(bw_grandstaff,detected_line)

    # Step 2: Smooth remaining noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    smoothed_image = cv2.morphologyEx(cleaned_horizontal, cv2.MORPH_CLOSE, closing_kernel)

    # Step 3: Detect Vertical line
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detected_vertical_lines = cv2.morphologyEx(smoothed_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Step 4: Erode to Isolate Note Heads
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,1))
    cleaned_image = cv2.erode(cleaned_horizontal, erode_kernel, iterations=1)  # Final erosion
    
    # Step 5: Smooth the note heads
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, closing_kernel,iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    cleaned_image = cv2.dilate(cleaned_image, dilate_kernel, iterations=1)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned_image = cv2.erode(cleaned_image, erode_kernel, iterations=1)  # Final erosion

    music_element_only_path = os.path.join(gs_path,"Only_Music_Element")
    os.makedirs(music_element_only_path,exist_ok=True)
    save_path = os.path.join(music_element_only_path,f"ME_of_GrandStaff_{gs_num+1}.png")
    cv2.imwrite(save_path,cleaned_image)
    print(f"Already, Process to have only music element")
    return cleaned_image, smoothed_image, detected_vertical_lines

def Detect_NoteHead(note_head_image, original_image,gs_num):
    contours, _ = cv2.findContours(note_head_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    note_heads = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        note_heads.append((x, y, w, h))  # Store note heads with coordinates

    # Sort note heads from left to right
    note_heads = sorted(note_heads, key=lambda b: b[0])

    # Separate into upper & lower staff using midpoint threshold
    staff_mid = np.median([y + h // 2 for _, y, _, h in note_heads])  # Estimate midpoint
    upper_notes = [n for n in note_heads if n[1] < staff_mid]
    lower_notes = [n for n in note_heads if n[1] >= staff_mid]

    # Convert image to color for annotation
    annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Annotate upper notes
    for i, (x, y, w, h) in enumerate(upper_notes):
        cv2.putText(annotated_image, str(i+1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Annotate lower notes
    for i, (x, y, w, h) in enumerate(lower_notes):
        cv2.putText(annotated_image, str(i+1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save Directory
    save_dir = "Twinkle_Twinkle_Little_Star/Detected_NoteHead"
    os.makedirs(save_dir, exist_ok=True)

    # Save and return annotated image
    annotated_output_path = os.path.join(save_dir, f"annotated_note_heads_{gs_num+1}.png")
    cv2.imwrite(annotated_output_path, annotated_image)
    return annotated_image, upper_notes, lower_notes


def main(gs_path):
    grand_staffs = [g for g in os.listdir(gs_path) if g.lower().endswith("png")]
    for gs_num, grand_staff in enumerate(grand_staffs):
        upper_staff_boxes, lower_staff_boxes, annotated_path, detected_lines, bw_grandstaff = Line_Position(gs_path,grand_staff,gs_num)
        print(f"Upper_Staff_{gs_num+1}",upper_staff_boxes) #x, y, w, h
        print(f"LowerStaff_{gs_num+1}",lower_staff_boxes) #x, y, w, h
        note_head_image, music_element_only_image, detected_vertical_lines = Music_Element(gs_path,gs_num,detected_lines,bw_grandstaff)
        annotated_image, upper_notes, lower_notes = Detect_NoteHead(note_head_image, bw_grandstaff,gs_num)
        print("************************************************")
        
if __name__ == "__main__":
    GRAND_STAFF_PATH = "Twinkle_Twinkle_Little_Star\Grand_Staff_IMG"
    main(GRAND_STAFF_PATH)
