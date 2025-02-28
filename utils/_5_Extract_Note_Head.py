"""
Input:
* Object position 

Output:
* 
"""
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper")

        # Select folders
        self.source_folder = filedialog.askdirectory(title="Select Source Folder")
        self.save_folder = filedialog.askdirectory(title="Select Save Folder")
        
        if not self.source_folder or not self.save_folder:
            print("Folders not selected! Exiting.")
            self.root.quit()
            return

        self.image_files = [f for f in os.listdir(self.source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.current_index = 0

        # Create UI
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.load_image()
        
        # Mouse bindings for cropping
        self.canvas.bind("<ButtonPress-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.draw_crop_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_crop)
        
        # Navigation Buttons
        self.btn_next = tk.Button(root, text="Next Image", command=self.next_image)
        self.btn_next.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.btn_prev = tk.Button(root, text="Previous Image", command=self.prev_image)
        self.btn_prev.pack(side=tk.RIGHT, padx=10, pady=5)

        self.crop_box = None

    def load_image(self):
        if self.current_index >= len(self.image_files):
            print("No more images.")
            return
        
        image_path = os.path.join(self.source_folder, self.image_files[self.current_index])
        self.cv_image = cv2.imread(image_path)
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.cv_image)
        
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.config(width=self.pil_image.width, height=self.pil_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def start_crop(self, event):
        self.crop_box = [event.x, event.y, event.x, event.y]

    def draw_crop_box(self, event):
        if self.crop_box:
            self.crop_box[2], self.crop_box[3] = event.x, event.y
            self.canvas.delete("crop_rect")
            self.canvas.create_rectangle(*self.crop_box, outline="red", width=2, tags="crop_rect")

    def finish_crop(self, event):
        if not self.crop_box:
            return
        x1, y1, x2, y2 = self.crop_box
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        cropped_img = self.cv_image[y1:y2, x1:x2]
        save_path = os.path.join(self.save_folder, f"cropped_{self.image_files[self.current_index]}")
        cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        print(f"Saved cropped image: {save_path}")
        self.crop_box = None

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
