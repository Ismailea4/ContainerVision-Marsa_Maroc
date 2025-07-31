import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import os
import numpy as np
import sys
from io import StringIO
import time


def check_digit_verification(code):
    """
    Validates a code by checking if its check digit matches a calculated value.

    The function assigns numerical values to letters, verifies the code format,
    and computes a weighted sum to compare against the check digit.

    Args:
        code (dict): Dictionary containing a 'CN' key with an 11-character string.

    Returns:
        bool: True if the check digit is valid, False otherwise.
    """
    # Create numerical value assigned to each letter of the alphabet using code ascii
    cst = 10
    letter_values = {"A":10, "B":12, "C":13, "D":14, "E":15, "F":16, "G":17, "H":18, "I":19,
                     "J":20, "K":21, "L":23, "M":24, "N":25, "O":26, "P":27, "Q":28, "R":29,
                     "S":30, "T":31, "U":32, "V":34, "W":35, "X":36, "Y":37, "Z":38}
    """
    for i in range(65, 91):
        if (i - 65 + cst) % 11 == 0:
            cst += 1 
        letter_values[chr(i)] = i - 65 + cst  
    """
    
    
    # Verify if the first 4 are letters and the last 7 are digits
    if len(code) == 11 and code[:4].isalpha() and code[4:].isdigit():
        check_digit = int(code[-1])
        i = 0
        sum = 0
        for char in code[:-1]:
            if char.isalpha():
                sum += letter_values[char] * 2**i
            else:
                sum += int(char) * 2**i
            i += 1
        if sum % 11 == 10:
            sum = 0
    
        return check_digit == sum % 11
    return False

def code_verification_info(code):
    """
    Verify the extracted code.
    
    Args:
        code (dict): Dictionary containing the extracted codes.
        
    Returns:
        str: The information about the validation of the code.
    """
    CN = code.get('CN', {}).get('value', '')
    TS = code.get('TS', {}).get('value', '')
    info = "Verification Results:\n"
    # Example verification logic (to be replaced with actual logic)
    if CN:
        info += f"Container Number: {CN}\n"
        if len(CN) == 11:
            if not check_digit_verification(CN):
                info += f"/!\ Invalid Container Number (check digit mismatch) .\n"
            else:
                info += f"Valid Container Number !\n"
        else:
            info += f"/!\ Invalid Container Number length.\n"
            
    if TS:
        info += f"ISO Code: {TS}\n"
        if len(TS) == 4:
            info += f"Valid ISO Code !\n"
        else:
            info += f"/!\ Invalid ISO Code length.\n"
    return info

class ContainerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Container Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        self.fullscreen_window = None
        
        # Bind F11 for fullscreen toggle
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        # Variables
        self.current_image_path = None
        self.processed_image = None
        self.original_image = None
        self.terminal_output = StringIO()
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Container Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel for controls
        self.setup_left_panel(main_frame)
        
        # Middle panel for image display
        self.setup_middle_panel(main_frame)
        
        # Right panel for predictions
        self.setup_right_panel(main_frame)
        
        # Bottom panel for progress and terminal output
        self.setup_bottom_panel(main_frame)
        
    def setup_left_panel(self, parent):
        # Controls frame
        controls_frame = ttk.LabelFrame(parent, text="Controls & Results", padding="10")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        controls_frame.columnconfigure(1, weight=1)
        
        # Container Number
        ttk.Label(controls_frame, text="Container Number:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.container_number_var = tk.StringVar(value="--")
        ttk.Label(controls_frame, textvariable=self.container_number_var, 
                 font=("Arial", 10), foreground="blue").grid(
            row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # ISO Code
        ttk.Label(controls_frame, text="ISO Code:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.iso_code_var = tk.StringVar(value="--")
        ttk.Label(controls_frame, textvariable=self.iso_code_var, 
                 font=("Arial", 10), foreground="blue").grid(
            row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Sealed Count
        ttk.Label(controls_frame, text="Sealed Count:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=5)
        self.sealed_count_var = tk.StringVar(value="--")
        ttk.Label(controls_frame, textvariable=self.sealed_count_var, 
                 font=("Arial", 10), foreground="green").grid(
            row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Unsealed Count
        ttk.Label(controls_frame, text="Unsealed Count:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        self.unsealed_count_var = tk.StringVar(value="--")
        ttk.Label(controls_frame, textvariable=self.unsealed_count_var, 
                 font=("Arial", 10), foreground="red").grid(
            row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Separator
        ttk.Separator(controls_frame, orient='horizontal').grid(
            row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        
        # Upload button
        self.upload_btn = ttk.Button(controls_frame, text="Select Image", 
                                   command=self.upload_image, style="Accent.TButton")
        self.upload_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Process button
        self.process_btn = ttk.Button(controls_frame, text="Start Detection", 
                                    command=self.start_detection, state="disabled")
        self.process_btn.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Save button
        self.save_btn = ttk.Button(controls_frame, text="Save Output Image", 
                                 command=self.save_output_image, state="disabled")
        self.save_btn.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Clear button
        self.clear_btn = ttk.Button(controls_frame, text="Clear Results", 
                                  command=self.clear_results)
        self.clear_btn.grid(row=8, column=0, columnspan=2, pady=5)
        
        # Terminal output frame
        terminal_frame = ttk.LabelFrame(controls_frame, text="Processing Log", padding="10")
        terminal_frame.grid(row=9, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        terminal_frame.columnconfigure(0, weight=0)
        terminal_frame.rowconfigure(0, weight=0)
        
        # Text widget for terminal output
        self.terminal_text = scrolledtext.ScrolledText(
            terminal_frame,
            wrap=tk.WORD,
            width=30,
            height=10,
            font=("Consolas", 9),
            bg="black",
            fg="lightgreen",
            insertbackground="white"
        )
        self.terminal_text.grid(row=9, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.terminal_text.insert(tk.END, "Processing log will appear here...\n")
        self.terminal_text.config(state=tk.DISABLED)
        
    def setup_middle_panel(self, parent):
        # Image display frame
        image_frame = ttk.LabelFrame(parent, text="Image Display", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="white", relief="sunken", borderwidth=2)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add Fullscreen button
        fullscreen_btn = ttk.Button(image_frame, text="Toggle Fullscreen (F11)", command=self.toggle_fullscreen)
        fullscreen_btn.grid(row=2, column=0, pady=(10, 0), sticky=tk.W)

        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)

        # Bind mouse click to focus canvas for keyboard events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.focus_set()  # To receive key events

        # Default message
        self.canvas.create_text(
            400, 300,
            text="No image selected\nClick 'Select Image' to begin",
            font=("Arial", 14), fill="gray",
            justify="center"
        )
    
    def setup_right_panel(self, parent):
        # Predictions frame
        predictions_frame = ttk.LabelFrame(parent, text="Prediction Output", padding="10")
        predictions_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        predictions_frame.columnconfigure(0, weight=1)
        predictions_frame.rowconfigure(0, weight=1)
        
        # Text widget for predictions with scrollbar
        self.predictions_text = scrolledtext.ScrolledText(
            predictions_frame, 
            wrap=tk.WORD, 
            width=30, 
            height=20,
            font=("Consolas", 10),
            bg="white",
            fg="black",
            selectbackground="lightblue"
        )
        self.predictions_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial message
        self.predictions_text.insert(tk.END, "Prediction output will appear here after detection...\n\n")
        self.predictions_text.insert(tk.END, "This text can be selected and copied using Ctrl+C.")
        self.predictions_text.config(state=tk.DISABLED)  # Make it read-only initially
        
        # Button to copy predictions
        copy_btn = ttk.Button(predictions_frame, text="Copy Predictions", 
                             command=self.copy_predictions)
        copy_btn.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)
        
    def setup_bottom_panel(self, parent):
        # Bottom frame for progress and terminal output
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(1, weight=1)
        
        # Progress label and bar
        progress_info_frame = ttk.Frame(bottom_frame)
        progress_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_info_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_info_frame, textvariable=self.progress_var, 
                                      font=("Arial", 10))
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_info_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
    def on_canvas_click(self, event):
        """Handle canvas click to focus for keyboard events"""
        self.canvas.focus_set()
            
    def toggle_fullscreen(self, event=None):
        if self.fullscreen_window is None:
            # Create a new fullscreen window for the image
            self.fullscreen_window = tk.Toplevel(self.root)
            self.fullscreen_window.attributes("-fullscreen", True)
            self.fullscreen_window.configure(bg="black")
            self.fullscreen_window.bind("<Escape>", self.exit_fullscreen)
            self.fullscreen_window.bind("<Button-1>", self.exit_fullscreen)

            # Display the image in fullscreen
            if self.original_image:
                screen_width = self.fullscreen_window.winfo_screenwidth()
                screen_height = self.fullscreen_window.winfo_screenheight()
                img = self.original_image.copy()
                img_ratio = img.width / img.height
                screen_ratio = screen_width / screen_height

                if img_ratio > screen_ratio:
                    new_width = screen_width
                    new_height = int(screen_width / img_ratio)
                else:
                    new_height = screen_height
                    new_width = int(screen_height * img_ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.fullscreen_photo = ImageTk.PhotoImage(img)
                canvas = tk.Canvas(self.fullscreen_window, bg="black", highlightthickness=0)
                canvas.pack(fill="both", expand=True)
                canvas.create_image(screen_width//2, screen_height//2, image=self.fullscreen_photo, anchor="center")
                canvas.bind("<Escape>", self.exit_fullscreen)
                canvas.bind("<Button-1>", self.exit_fullscreen)
        else:
            self.exit_fullscreen()

    def exit_fullscreen(self, event=None):
        if self.fullscreen_window is not None:
            self.fullscreen_window.destroy()
            self.fullscreen_window = None

    def update_image_display(self):
        """Update the image display (fit to canvas)"""
        if not self.original_image:
            return

        try:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 400

            original_width, original_height = self.original_image.size
            scale = min(canvas_width/original_width, canvas_height/original_height, 1.0)
            display_width = int(original_width * scale)
            display_height = int(original_height * scale)

            # Resize image
            display_image = self.original_image.resize(
                (display_width, display_height), 
                Image.Resampling.LANCZOS
            )

            # Update PhotoImage
            self.photo = ImageTk.PhotoImage(display_image)

            # Clear canvas and display image centered
            self.canvas.delete("all")
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            self.canvas.create_image(center_x, center_y, image=self.photo, anchor="center")

            # Update scroll region
            scroll_width = max(display_width, canvas_width)
            scroll_height = max(display_height, canvas_height)
            self.canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update image display: {str(e)}")

    def save_output_image(self):
        """Save the processed output image"""
        if not self.processed_image:
            messagebox.showwarning("Warning", "No processed image to save!")
            return
        
        file_types = [
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.asksaveasfilename(
            title="Save Output Image",
            defaultextension=".png",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                
    

    def log_message(self, message):
        """
        #Add a message to the terminal output
        """

        self.terminal_text.config(state=tk.NORMAL)
        self.terminal_text.insert(tk.END, f"{message}\n")
        self.terminal_text.see(tk.END)  # Auto-scroll to bottom
        self.terminal_text.config(state=tk.DISABLED)
        self.root.update_idletasks()  # Update UI immediately

    def copy_predictions(self):
        """Copy predictions text to clipboard"""
        try:
            predictions_content = self.predictions_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(predictions_content)
            self.log_message("Predictions copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy predictions: {str(e)}")
        
    def upload_image(self):
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.gif'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Container Image",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.process_btn.config(state="normal")
            self.progress_var.set(f"Image loaded: {os.path.basename(file_path)}")
            self.log_message(f"Image loaded: {os.path.basename(file_path)}")
            
    def display_image(self, image_path):
        try:
            # Open image and store as original
            self.original_image = Image.open(image_path)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')

            # Update display
            self.update_image_display()
            self.log_message("Image displayed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.log_message(f"Error loading image: {str(e)}")
            
    def start_detection(self):
        self.time = time.time()  # Start timer for detection
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        # Clear previous results
        self.predictions_text.config(state=tk.NORMAL)
        self.predictions_text.delete(1.0, tk.END)
        self.predictions_text.config(state=tk.DISABLED)

        self.terminal_text.config(state=tk.NORMAL)
        self.terminal_text.delete(1.0, tk.END)
        self.terminal_text.config(state=tk.DISABLED)

        # Start detection in a separate thread to prevent UI freezing
        self.process_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.progress_bar.start()
        self.progress_var.set("Running container detection...")
        self.log_message("Starting container detection...")
        
        detection_thread = threading.Thread(target=self.run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
    def run_detection(self):
        try:
            self.log_message("Initializing detection model...")
            
            # Import your detection function here
            from my_detection_module import container_detection
            
            self.log_message("Processing image...")
            
            # For now, using simulated results - replace this with your actual function
            output = container_detection(self.current_image_path)
            
            # Simulate the actual detection process
            import time
            self.log_message("Loading AI models...")
            self.log_message("Analyzing container features...")
            self.log_message("Detecting text regions...")
            self.log_message("Running OCR on detected regions...")
            
            
            
            results = output['detections']
            processed_image = output['predictions']
            
            self.log_message("Detection completed successfully!")
            
            # Update UI in main thread
            self.root.after(0, self.update_results, results, processed_image)
            
        except Exception as e:
            self.log_message(f"Error during detection: {str(e)}")
            self.root.after(0, self.detection_error, str(e))
            
    def update_results(self, results, processed_image):
        # Update result variables
        self.log_message("Updating results...")
        
        self.container_number_var.set(results.get('CN', {}).get('value','--'))
        self.iso_code_var.set(results.get('TS', {}).get('value','--'))
        self.sealed_count_var.set(str(results.get('sealed', {}).get('value', 0)))
        self.unsealed_count_var.set(str(results.get('unsealed', {}).get('value', 0)))

        # Update predictions text widget
        self.predictions_text.config(state=tk.NORMAL)
        self.predictions_text.delete(1.0, tk.END)
        
        # Handle processed image
        if isinstance(processed_image, np.ndarray):
            import cv2
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(processed_image)

        if isinstance(processed_image, Image.Image):
            processed_image = processed_image.convert("RGB")
            self.processed_image = processed_image
        
        # Format predictions output
        predictions_output = "=== CONTAINER DETECTION RESULTS ===\n\n"
        predictions_output += f"Container Number: {results.get('CN', {}).get('value', 'Not detected')}\n"
        predictions_output += f"  Confidence: {results.get('CN', {}).get('confidence', 0):.2%}\n\n"
        
        predictions_output += f"ISO Code: {results.get('TS', {}).get('value', 'Not detected')}\n"
        predictions_output += f"  Confidence: {results.get('TS', {}).get('confidence', 0):.2%}\n\n"
        
        predictions_output += f"Sealed Containers: {results.get('sealed', {}).get('value', 0)}\n"
        predictions_output += f"  Confidence: {results.get('sealed', {}).get('confidence', 0):.2%}\n\n"
        
        predictions_output += f"Unsealed Containers: {results.get('unsealed', {}).get('value', 0)}\n"
        predictions_output += f"  Confidence: {results.get('unsealed', {}).get('confidence', 0):.2%}\n\n"
        
        predictions_output += "=== CHECK DETAILS ===\n"
        check_info = code_verification_info(results)
        predictions_output += check_info + "\n"
        
        predictions_output += "=== ADDITIONAL DETAILS ===\n"
        predictions_output += f"Processing Time: {(time.time() - self.time):.1f} seconds\n"
        predictions_output += f"Image Size: {processed_image.size[0]} x {processed_image.size[1]} pixels\n"
        predictions_output += f"Detection Model: YOLOv8-Container\n"
        predictions_output += f"OCR Model: ResNetChar \n\n"
        
        predictions_output += "This output can be selected and copied using Ctrl+C or the Copy button below."
        
        self.predictions_text.insert(tk.END, predictions_output)
        self.predictions_text.config(state=tk.DISABLED)

        # Store original image for display and display processed image
        self.original_image = processed_image
        self.update_image_display()
        self.log_message("Processed image displayed")
        
        self.progress_bar.stop()
        self.progress_var.set("Detection completed successfully!")
        self.process_btn.config(state="normal")
        self.save_btn.config(state="normal")  # Enable save button
        self.log_message("Results updated successfully.")
                
    def display_processed_image(self, processed_image):
        try:
            # Calculate display size while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 400
            
            img_width, img_height = processed_image.size
            scale = min(canvas_width/img_width, canvas_height/img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = processed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display processed image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.photo, anchor="center")
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.log_message("Processed image displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display processed image: {str(e)}")
            self.log_message(f"Error displaying processed image: {str(e)}")
            
    def detection_error(self, error_message):
        self.progress_bar.stop()
        self.progress_var.set("Detection failed!")
        self.process_btn.config(state="normal")
        messagebox.showerror("Detection Error", f"Detection failed: {error_message}")
        
    def clear_results(self):
        # Clear all results
        self.container_number_var.set("--")
        self.iso_code_var.set("--")
        self.sealed_count_var.set("--")
        self.unsealed_count_var.set("--")
        
        # Clear predictions
        self.predictions_text.config(state=tk.NORMAL)
        self.predictions_text.delete(1.0, tk.END)
        self.predictions_text.insert(tk.END, "Prediction output will appear here after detection...\n\n")
        self.predictions_text.insert(tk.END, "This text can be selected and copied using Ctrl+C.")
        self.predictions_text.config(state=tk.DISABLED)

        # Clear terminal
        self.terminal_text.config(state=tk.NORMAL)
        self.terminal_text.delete(1.0, tk.END)
        self.terminal_text.insert(tk.END, "Processing log will appear here...\n")
        self.terminal_text.config(state=tk.DISABLED)

        # Clear image
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="No image selected\nClick 'Select Image' to begin", 
                              font=("Arial", 14), fill="gray", justify="center")
        
        # Reset image variables
        self.original_image = None
        self.processed_image = None

        # Reset progress
        self.progress_var.set("Ready")
        self.current_image_path = None
        self.process_btn.config(state="disabled")
        self.save_btn.config(state="disabled")

def main():
    root = tk.Tk()
    app = ContainerDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()