"""
1.Read file name then go into the note_project floder
2.For each page extract the grand_staff into the grand_staff floder, then save as grad_staff_number 
"""
"""        if "Grand_Staffs" in os.listdir(project_floder):
            print(f"Grand_Staffs of {project_floder} already Exist")
            return 1
        output_folder = f"{project_floder}/Grand_Staffs"
        os.makedirs(output_folder, exist_ok=True)
        binary.save(output_folder, "PNG")
        print("Grand_Staffs are extracted successfully")
    return 1
"""
import cv2
import numpy as np
import os

def Line_Detection(project_folder, page, page_num):
    image_path = os.path.join(project_folder, page)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    # Step 1: Binary thresholding
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        
    # Step 2: Detect horizontal lines (staff lines)
    kernel_width = max(image.shape[1] // 30, 1)  # ensure kernel width is at least 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    print(f"Page {page_num+1} has extracted the lines successfully")
    return detected_lines

def Grand_Staff_Extraction(project_folder, page, detected_lines, page_num):
    # Step 1: Find contours of staff lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])  # sort by vertical (y) position

    # Step 2: Calculate line spacing dynamically for expanding bounding boxes
    if len(bounding_boxes) < 2:
        print(f"Not enough bounding boxes on page {page_num+1} to calculate line spacing.")
        return []
    line_spacing = np.median([bounding_boxes[i + 1][1] - bounding_boxes[i][1] 
                              for i in range(len(bounding_boxes) - 1)])

    # Step 3: Group bounding boxes into sets of 10 lines (2 staves per grand staff)
    width = detected_lines.shape[1]
    grouped_boxes = []
    for i in range(0, len(bounding_boxes), 10):
        group = bounding_boxes[i:i + 10]
        if group:
            x_min = 0
            y_min = min([box[1] for box in group]) - int(line_spacing * 2)
            x_max = width
            y_max = max([box[1] + box[3] for box in group]) + int(line_spacing * 2)
            grouped_boxes.append((x_min, y_min, x_max, y_max))

    # Step 4: Draw rectangles and annotate grand staff regions
    image_path = os.path.join(project_folder, page)  # use the page filename!
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error loading original image for annotation: {image_path}")
        return []
    image_with_boxes = original_image.copy()
    for idx, (x_min, y_min, x_max, y_max) in enumerate(grouped_boxes):
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f"Grand Staff {idx + 1}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(project_folder, f"Boxed_Image_{page_num+1}.png"), image_with_boxes)

    # Step 5: Crop grand staff regions and return them
    grand_staff_images = []
    for idx, (x_min, y_min, x_max, y_max) in enumerate(grouped_boxes):
        cropped_region = original_image[y_min:y_max, x_min:x_max]
        grand_staff_images.append(cropped_region)
    return grand_staff_images

def Save_Grand_Staffs(project_folder, all_grand_staff):
    gs_folder_path = os.path.join(project_folder, "Grand_Staff_IMG")
    os.makedirs(gs_folder_path, exist_ok=True)
    for i, grandstaff in enumerate(all_grand_staff):
        save_path = os.path.join(gs_folder_path, f"grand_staff_{i+1}.png")
        success = cv2.imwrite(save_path, grandstaff)
        if success:
            print(f"Saved {save_path} successfully.")
        else:
            print(f"Failed to save {save_path}.")
    return gs_folder_path

def main(target_folder):
    all_grand_staff = []
    pages = [p for p in os.listdir(target_folder) if p.lower().endswith('.png') and p[0:4]=="page"]
    print(pages)
    for i, page in enumerate(pages):
        detected_line_img = Line_Detection(target_folder, page, i)
        if detected_line_img is None:
            continue
        grandstaffs_onePage_list = Grand_Staff_Extraction(target_folder, page, detected_line_img, i)
        all_grand_staff.extend(grandstaffs_onePage_list)
    GRAND_STAFF_PATH = Save_Grand_Staffs(target_folder, all_grand_staff)

if __name__ == "__main__":
    project_folder = "Twinkle_Twinkle_Little_Star"
    main(project_folder)
