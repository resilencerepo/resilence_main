# main.py
import os
import cv2
import random
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import torch
import math
try:
    from SonicSurface.ControlSoftware.Python.SonicSurface import SonicSurface
    from SonicSurface.ControlSoftware.Python.frametimer import FrameTimer
    sonic_surface_available = True
except ImportError as e:
    print(f"Failed to import SonicSurface or FrameTimer: {e}")
    sonic_surface_available = False

# Assuming these are in your project directory
from detector import PersonDetector
from feature_extractor import FeatureExtractor
from story_generator import StoryGenerator
from tts import TextToSpeechClean

from dotenv import load_dotenv
load_dotenv()

# Ensure GPU is used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables for components
detector = None
feature_extractor = None
story_gen = None
tts = None
sonic_surface = None

# Queue for inter-thread communication
person_data_queue = queue.Queue()
feature_extraction_queue = queue.Queue()

# Flags to control thread execution
running = True
story_processing_active = False
last_story_completion = 0
STORY_INTERVAL = 60  # seconds between story generations

# SonicSurface constants
ANGLE_MAX = 20 * (np.pi/180)  # Max steering angle (20 degrees)
MOD_FREQ = 600  # Increased for better modulation
TIME_PER_POS = 0.1  # Increased to 0.1s for longer tone duration
WAIT_SWITCH = 1.0 / MOD_FREQ / 2
N_SWITCHES = int(TIME_PER_POS / WAIT_SWITCH)
SWEEP_SPEED = 0.5 * (np.pi/180)  # Angular speed for beam sweeping

def initialize_components():
    """Initializes all components once."""
    global detector, feature_extractor, story_gen, tts, sonic_surface
    print("Initializing components...")
    detector = PersonDetector(model_name=os.getenv("DETECTOR_MODEL", 'yolov8n.pt'))
    feature_extractor = FeatureExtractor()
    story_gen = StoryGenerator(model="phi3.5")
    tts = TextToSpeechClean()
    if sonic_surface_available:
        try:
            sonic_surface = SonicSurface()
            sonic_surface.connect(-1)  # Allow user to select serial port
            print("SonicSurface initialized and connected.")
        except Exception as e:
            print(f"Error initializing SonicSurface: {e}")
            sonic_surface = None
    else:
        print("SonicSurface not available due to import error.")
        sonic_surface = None
    print("Components initialized.")

class Surface:
    def __init__(self, sonic_surface):
        self.sonic_surface = sonic_surface
        self.commit_thread = None
        self.running = False
        self.ticker = FrameTimer(500)

    def start_sweeping(self):
        """Start sweeping the beam with melody."""
        self.commit_thread = threading.Thread(target=self._sweeping_thread)
        self.running = True
        self.commit_thread.start()

    def start_targeting(self, target_angle):
        """Start targeting a fixed position with signal sound."""
        self.commit_thread = threading.Thread(target=self._targeting_thread, args=(target_angle,))
        self.running = True
        self.commit_thread.start()

    def stop(self):
        """Stop the ultrasonic beam and thread."""
        self.running = False
        if self.commit_thread:
            self.commit_thread.join()
        if self.sonic_surface:
            self.sonic_surface.switchOnOrOff(True)

    def _sweeping_thread(self):
        """Sweep beam from left to right and back with melody."""
        frequencies = np.array([220.00, 261.63, 329.63, 293.66, 349.23, 440.00, 329.63, 392.00, 493.88]) * 8
        angles = np.linspace(-ANGLE_MAX, ANGLE_MAX, 100)  # Right to left
        angle_index = 0
        direction = 1  # 1 for right, -1 for left
        fi = 0
        last_change = time.perf_counter()

        while self.running:
            self.ticker.tick()
            if self.sonic_surface:
                # Calculate current angle (reversed: positive angles to left)
                current_angle = -angles[angle_index]  # Reverse angle
                x = 5 * np.sin(current_angle)
                y = 5 * np.cos(current_angle)
                self.sonic_surface.focusAtPos(x, y, 0)
                self.sonic_surface.sendCommit()
                self.sonic_surface.switchOnOrOff(False)
                print(f"Sweeping - Angle: {current_angle * 180 / np.pi:.2f} degrees, Frequency: {frequencies[fi]:.2f} Hz")

                # Update angle for sweeping
                angle_index += direction
                if angle_index >= len(angles) - 1:
                    direction = -1  # Reverse to sweep right
                elif angle_index <= 0:
                    direction = 1   # Reverse to sweep left

                # Update frequency for melody
                now = time.perf_counter()
                if now - last_change >= TIME_PER_POS:
                    last_change = now
                    fi = (fi + 1) % len(frequencies)
                    self.ticker.target_fps = frequencies[fi]
                    self.sonic_surface.switchOnOrOff(False)

    def _targeting_thread(self, target_angle):
        """Focus beam on target with signal sound."""
        signal_frequencies = [2000, 2500]
        fi = 0
        last_change = time.perf_counter()
        x = 5 * np.sin(-target_angle)  # Reverse angle
        y = 5 * np.cos(-target_angle)
        target = (x, y, 0)

        while self.running:
            self.ticker.tick()
            if self.sonic_surface:
                self.sonic_surface.focusAtPos(*target)
                self.sonic_surface.sendCommit()
                self.sonic_surface.switchOnOrOff(False)
                print(f"Targeting - Angle: {-target_angle * 180 / np.pi:.2f} degrees, Frequency: {signal_frequencies[fi]:.2f} Hz")
                now = time.perf_counter()
                if now - last_change >= TIME_PER_POS:
                    last_change = now
                    fi = (fi + 1) % len(signal_frequencies)
                    self.ticker.target_fps = signal_frequencies[fi]
                    self.sonic_surface.switchOnOrOff(False)

def update_frame(panel, frame):
    """Updates the Tkinter Label with the latest video frame with corrected colors."""
    try:
        # Fix the blue/purple color cast by adjusting white balance
        # Convert to LAB color space for better color correction
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab)
        
        # Reduce the blue cast by adjusting the b channel (blue-yellow axis)
        # Negative values in b channel indicate blue cast
        b = cv2.addWeighted(b, 0.85, np.full_like(b, 128), 0.15, 0)
        
        # Slightly adjust the a channel (green-red axis) for better balance  
        a = cv2.addWeighted(a, 0.95, np.full_like(a, 128), 0.05, 0)
        
        # Merge back and convert to BGR
        corrected_lab = cv2.merge([l, a, b])
        corrected_frame = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Convert to RGB for Tkinter
        img = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        panel.update_idletasks()
        panel_width = panel.winfo_width()
        panel_height = panel.winfo_height()
        
        if panel_width > 1 and panel_height > 1:
            img_ratio = img.width / img.height
            panel_ratio = panel_width / panel_height
            
            if img_ratio > panel_ratio:
                new_width = panel_width
                new_height = int(panel_width / img_ratio)
            else:
                new_height = panel_height
                new_width = int(panel_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    except Exception as e:
        print(f"Error updating frame: {e}")

        
def update_person_details(person_img, features, person_image_label, details_text_widget, target_angle, surface):
    """Updates the person details panel with new person information and steers ultrasonic beam."""
    try:
        # Update person image
        if person_img is not None:
            img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Resize while preserving aspect ratio to fit display area (300x400)
            display_width, display_height = 300, 400
            img_ratio = img.width / img.height
            if img_ratio > display_width / display_height:
                new_width = display_width
                new_height = int(display_width / img_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            person_image_label.imgtk = imgtk
            person_image_label.config(image=imgtk)
        
        # Update details text with status
        details_text_widget.config(state="normal")
        details_text_widget.delete("1.0", tk.END)
        
        details_text = "=== DETECTED PERSON DETAILS ===\n\n"
        details_text += f"AGE: {features.get('age', 'Unknown')}\n"
        details_text += f"GENDER: {features.get('gender', 'Unknown').upper()}\n"
        details_text += f"EMOTION: {features.get('emotion', 'Unknown').upper()}\n"
        details_text += f"STYLE: {features.get('style', 'Unknown').upper()}\n\n"
        clothing_items = features.get('clothing', [])
        if clothing_items:
            details_text += "CLOTHING DETECTED:\n"
            for item in clothing_items:
                details_text += f"• {item.upper()}\n"
        else:
            details_text += "CLOTHING DETECTED:\n• NO SPECIFIC ITEMS DETECTED\n"
        details_text += "\n" + "="*40 + "\n"
        details_text += "STATUS: GENERATING STORY AND SPEAKING...\n"
        
        details_text_widget.insert(tk.END, details_text)
        details_text_widget.config(state="disabled")

        # Stop SonicSurface during story generation and playback
        if surface:
            surface.stop()

        # Generate and speak story in a separate thread
        def generate_and_speak_story():
            global story_processing_active, last_story_completion
            try:
                print("Starting story generation...")
                story = story_gen.generate_story(features)
                print(f"Generated Story: {story}")
                
                print("Starting speech...")
                tts.speak(story)
                print("Speech finished.")
                
                # Update completion time and status
                last_story_completion = time.time()
                story_processing_active = False
                
                # Clear image and features, revert to TRACKING IN PROCESS
                person_image_label.after(0, lambda: person_image_label.config(image='', text="TRACKING IN PROCESS", font=('Arial', 12)))
                details_text_widget.after(0, lambda: details_text_widget.config(state="normal"))
                details_text_widget.after(0, lambda: details_text_widget.delete("1.0", tk.END))
                details_text_widget.after(0, lambda: details_text_widget.insert(tk.END, "TRACKING IN PROCESS"))
                details_text_widget.after(0, lambda: details_text_widget.config(state="disabled"))
                
                # Restart sweeping after story
                if surface:
                    surface.start_sweeping()
                
            except Exception as e:
                print(f"Error in story generation or TTS: {e}")
                story_processing_active = False
                # Clear image and features, show error
                person_image_label.after(0, lambda: person_image_label.config(image='', text="TRACKING IN PROCESS", font=('Arial', 12)))
                details_text_widget.after(0, lambda: details_text_widget.config(state="normal"))
                details_text_widget.after(0, lambda: details_text_widget.delete("1.0", tk.END))
                details_text_widget.after(0, lambda: details_text_widget.insert(tk.END, "TRACKING IN PROCESS"))
                details_text_widget.after(0, lambda: details_text_widget.config(state="disabled"))
                # Restart sweeping
                if surface:
                    surface.start_sweeping()

        threading.Thread(target=generate_and_speak_story, daemon=True).start()

    except Exception as e:
        print(f"Error updating person details: {e}")
        # Clear image and features on error
        person_image_label.config(image='', text="TRACKING IN PROCESS", font=('Arial', 12))
        details_text_widget.config(state="normal")
        details_text_widget.delete("1.0", tk.END)
        details_text_widget.insert(tk.END, "TRACKING IN PROCESS")
        details_text_widget.config(state="disabled")
        # Restart sweeping
        if surface:
            surface.start_sweeping()

def feature_extraction_thread():
    """Dedicated thread for feature extraction to avoid blocking video feed."""
    global feature_extractor, running
    
    while running:
        try:
            # Wait for person data from the queue
            person_data = feature_extraction_queue.get(timeout=1)
            if person_data is None:  # Shutdown signal
                break
                
            person_img, bbox, target_angle, callback = person_data
            
            print("Extracting features in background...")
            features = feature_extractor.extract(person_img)
            features["person_bbox"] = bbox
            
            # Execute callback with results
            if callback:
                callback(person_img, features, target_angle)
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Feature extraction thread error: {e}")
    
    print("Feature extraction thread stopped.")

def video_processing_thread(video_source, live_tracking_panel, person_image_label, details_text_widget, root):
    """
    Thread for video capture and detection. Feature extraction moved to separate thread.
    """
    global running, story_processing_active, last_story_completion
    
    # Start sweeping during tracking
    surface = None
    if sonic_surface:
        surface = Surface(sonic_surface)
        surface.start_sweeping()

    while running:
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print("Error: Could not open video source")
                messagebox.showerror("Video Error", "Could not open video source.")
                time.sleep(5)
                continue

            # Initialize timing variables
            last_person_selection_time = 0

            while running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, reinitializing capture...")
                    break

                current_time = time.time()

                # Detect and track people (this is fast)
                annotated_frame = frame.copy()
                track_ids = []
                try:
                    annotated_frame, track_ids = detector.detect_and_track(frame)
                except Exception as e:
                    print(f"Detection error: {e}")
                    continue

                # Always update the live tracking panel immediately
                root.after(0, update_frame, live_tracking_panel, annotated_frame)

                # Update right panel to TRACKING IN PROCESS if no person is being processed
                if not story_processing_active and (current_time - last_story_completion >= STORY_INTERVAL):
                    root.after(0, lambda: person_image_label.config(image='', text="TRACKING IN PROCESS", font=('Arial', 12)))
                    root.after(0, lambda: details_text_widget.config(state="normal"))
                    root.after(0, lambda: details_text_widget.delete("1.0", tk.END))
                    root.after(0, lambda: details_text_widget.insert(tk.END, "TRACKING IN PROCESS"))
                    root.after(0, lambda: details_text_widget.config(state="disabled"))
                    # Ensure sweeping is active
                    if surface and not surface.running:
                        surface.start_sweeping()

                # Check if it's time to select a new person
                time_since_last_selection = current_time - last_person_selection_time
                time_since_last_story = current_time - last_story_completion
                
                should_select_person = (
                    time_since_last_selection >= STORY_INTERVAL and 
                    time_since_last_story >= STORY_INTERVAL and 
                    not story_processing_active and 
                    len(track_ids) > 0
                )

                if should_select_person:
                    last_person_selection_time = current_time
                    story_processing_active = True
                    
                    try:
                        # Quick person extraction
                        person_img, bbox = detector.get_random_person(frame, track_ids)
                        if person_img is None or person_img.size == 0:
                            print("No valid person image extracted")
                            story_processing_active = False
                            # Restart sweeping
                            if surface:
                                surface.start_sweeping()
                            continue

                        # Calculate steering angle from bounding box (reversed)
                        frame_width = frame.shape[1]
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        # Reverse mapping: left (0) to +ANGLE_MAX, right (width) to -ANGLE_MAX
                        target_angle = -((center_x / frame_width) - 0.5) * 2 * ANGLE_MAX

                        # Define callback for when features are ready
                        def feature_callback(img, features, angle):
                            root.after(0, update_person_details, img, features, 
                                     person_image_label, details_text_widget, angle, surface)

                        # Queue feature extraction with target angle
                        feature_extraction_queue.put((person_img, bbox, target_angle, feature_callback))
                        print("Person queued for feature extraction")

                    except Exception as e:
                        print(f"Person selection error: {e}")
                        story_processing_active = False
                        # Clear right panel and restart sweeping
                        root.after(0, lambda: person_image_label.config(image='', text="TRACKING IN PROCESS", font=('Arial', 12)))
                        root.after(0, lambda: details_text_widget.config(state="normal"))
                        root.after(0, lambda: details_text_widget.delete("1.0", tk.END))
                        root.after(0, lambda: details_text_widget.insert(tk.END, "TRACKING IN PROCESS"))
                        root.after(0, lambda: details_text_widget.config(state="disabled"))
                        if surface:
                            surface.start_sweeping()

                # Maintain ~30 FPS
                time.sleep(0.033)

            cap.release()
            
        except Exception as e:
            print(f"Major error in video thread: {e}")
            time.sleep(5)

    # Stop sweeping on shutdown
    if surface:
        surface.stop()
    print("Video processing thread stopped.")

def on_closing(root, video_thread, feature_thread):
    """Handles proper shutdown when the main window is closed."""
    global running, sonic_surface
    print("Closing application...")
    running = False
    
    # Signal feature extraction thread to stop
    feature_extraction_queue.put(None)
    
    # Wait for threads to finish
    if feature_thread.is_alive():
        feature_thread.join()
    if video_thread.is_alive():
        video_thread.join()
    
    # Disconnect SonicSurface
    if sonic_surface:
        sonic_surface.switchOnOrOff(True)
        sonic_surface.disconnect()
    
    cv2.destroyAllWindows()
    root.destroy()
    print("Application closed.")

def exit_fullscreen(event):
    """Exits fullscreen mode when ESC is pressed."""
    # Get the toplevel window (root) from any widget
    root = event.widget.winfo_toplevel()
    root.attributes('-fullscreen', False)
    root.attributes('-topmost', False)
    root.geometry("1280x720")
    root.update()

def main():
    initialize_components()

    # Create main window
    root = tk.Tk()
    root.title("Person Detection System")
    root.configure(bg='black')
    
    # Make window fullscreen and remove decorations
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate split dimensions
    left_width = screen_width // 2
    right_width = screen_width - left_width
    
    # Create main frame
    main_frame = tk.Frame(root, bg='black')
    main_frame.pack(fill='both', expand=True)
    
    # LEFT PANEL - Live Person Tracking
    left_frame = tk.Frame(main_frame, bg='black', width=left_width, height=screen_height)
    left_frame.pack(side='left', fill='both', expand=True)
    left_frame.pack_propagate(False)
    
    # Title for left panel
    left_title = tk.Label(left_frame, text="LIVE PERSON TRACKING", 
                         font=('Arial', 20, 'bold'), fg='white', bg='black')
    left_title.pack(pady=10)
    
    # Video display area
    live_tracking_panel = tk.Label(left_frame, bg='black', fg='white')
    live_tracking_panel.pack(padx=10, pady=10, fill='both', expand=True)
    
    # RIGHT PANEL - Selected Person Details
    right_frame = tk.Frame(main_frame, bg='black', width=right_width, height=screen_height)
    right_frame.pack(side='right', fill='both', expand=True)
    right_frame.pack_propagate(False)
    
    # Title for right panel
    right_title = tk.Label(right_frame, text="SELECTED PERSON DETAILS", 
                          font=('Arial', 20, 'bold'), fg='white', bg='black')
    right_title.pack(pady=10)
    
    # Person image display
    person_image_label = tk.Label(right_frame, bg='black', fg='white', 
                                 text="TRACKING IN PROCESS", 
                                 font=('Arial', 12))
    person_image_label.pack(pady=10)
    
    # Details text area
    details_frame = tk.Frame(right_frame, bg='black')
    details_frame.pack(padx=20, pady=10, fill='both', expand=True)
    
    details_text_widget = tk.Text(details_frame, wrap="word", 
                                 bg='black', fg='white', 
                                 font=('Courier', 12, 'bold'),
                                 insertbackground='white',
                                 selectbackground='gray',
                                 selectforeground='white',
                                 relief='flat',
                                 borderwidth=0)
    details_text_widget.pack(fill='both', expand=True)
    
    # Initial text
    details_text_widget.insert(tk.END, "TRACKING IN PROCESS")
    details_text_widget.config(state="disabled")
    
    # Add exit instruction
    exit_label = tk.Label(right_frame, text="Press ESC to exit fullscreen", 
                         font=('Arial', 10), fg='gray', bg='black')
    exit_label.pack(side='bottom', pady=5)
    
    # Bind ESC key to exit fullscreen
    root.bind('<Escape>', exit_fullscreen)
    
    # Start the feature extraction thread
    feature_thread = threading.Thread(target=feature_extraction_thread, daemon=True)
    feature_thread.start()
    
    # Start the video processing thread
    video_source = 0
    video_thread = threading.Thread(target=video_processing_thread,
                                   args=(video_source, live_tracking_panel, person_image_label, details_text_widget, root),
                                   daemon=True)
    video_thread.start()

    # Handle window closing gracefully
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, video_thread, feature_thread))

    root.mainloop()

if __name__ == "__main__":
    main()