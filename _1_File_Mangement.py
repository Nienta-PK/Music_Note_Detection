"""
1.Tkinter Dragdrop file into dataset (only pdf/png)
2.Select the file that user want in the combo_box
2.Check the dataset if it is png or pdf
    * if png then create note_project floder
    * if pdf then convert to png and save to note_project floder as page_number
"""

"""import os
import tkinter as tk
from tkinter import ttk, messagebox
from pdf2image import convert_from_path

# Poppler path (modify for your system)
POPPLER_PATH = r"C:\tools\poppler\Library\bin"  # Adjust this path if necessary
DATASET_FOLDER = "dataset"

# Function to get PDF files from the dataset folder
def get_pdf_files():
    return [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(".pdf")]

# Function to convert selected PDF to images
def convert_pdf(combo_box):
    selected_file = combo_box.get()
    if not selected_file:
        messagebox.showerror("Error", "Please select a PDF file!")
        return
    
    pdf_path = os.path.join(DATASET_FOLDER, selected_file)
    output_folder = os.path.splitext(selected_file)[0]
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        for i, img in enumerate(images):
            output_path = os.path.join(output_folder, f"page_{i+1}.png")
            img.save(output_path, "PNG")

        messagebox.showinfo("Success", f"PDF converted successfully! Images saved in '{output_folder}'")
    except Exception as e:
        messagebox.showerror("Error", f"Conversion failed: {e}")

if __name__ == '__main__':
    # Create GUI
    root = tk.Tk()
    root.title("PDF to PNG Converter")
    root.geometry("400x200")

    # Label
    label = tk.Label(root, text="Select a PDF file:")
    label.pack(pady=10)

    # ComboBox
    pdf_files = get_pdf_files()
    combo_box = ttk.Combobox(root, values=pdf_files, state="readonly")
    combo_box.pack(pady=5)
    if pdf_files:
        combo_box.current(0)  # Select first item by default

    # Convert Button
    convert_button = tk.Button(root, text="Convert", command=convert_pdf)
    convert_button.pack(pady=20)

    root.mainloop()"""

import os
from pdf2image import convert_from_path

def get_file(file_np,DATASET_FOLDER):
    files = os.listdir(DATASET_FOLDER)
    for file in files:
        file_name, suffix = file.split(".")
        if file_name == file_np:
            print(f"File Founded: {file_name}")
            return file,suffix
    print("File Doesn't Exist")
    return 0,0

def save_convert_file(file,suffix,DATASET_FOLDER,POPPLER_PATH):
    note_path = os.path.join(DATASET_FOLDER, file)
    output_folder = os.path.splitext(file)[0]
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) != 0:
        print("Note already Exist")
        return output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    if suffix == "png":
        output_path = os.path.join(output_folder, "page_1.png")
        img.save(output_path, "PNG")
        return output_path
    
    if suffix == "pdf":
        try:
            images = convert_from_path(note_path, dpi=300, poppler_path=POPPLER_PATH)
            for i, img in enumerate(images):
                output_path = os.path.join(output_folder, f"page_{i+1}.png")
                img.save(output_path, "PNG")

            print("Success", f"PDF converted successfully! Images saved in '{output_folder}'")
            return output_folder
        except Exception as e:
            print("Error", f"Conversion failed: {e}")
            return 0

def main(file_input,DATASET_FOLDER,POPPLER_PATH):
    file,suffix = get_file(file_input,DATASET_FOLDER)
    NOTE_PNG_PATH = save_convert_file(file,suffix,DATASET_FOLDER,POPPLER_PATH)


if __name__ == "__main__":
    file_input = "Twinkle_Twinkle_Little_Star"
    DATASET_FOLDER = "dataset"
    POPPLER_PATH = r"C:\tools\poppler\Library\bin"  # Adjust this path if necessary
    main(file_input,DATASET_FOLDER,POPPLER_PATH)