import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import threading
import math
import torch
import torchaudio
import json
# Import SAM Audio - handling potential import errors gracefully to allow GUI to open even if imports fail
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.config import SAMAudioConfig
    SAM_AVAILABLE = True
except ImportError as e:
    SAM_AVAILABLE = False
    print(f"Warning: Could not import sam_audio: {e}")

class SoundIsolatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Isolator")
        self.root.geometry("600x450")
        
        # Styles
        self.bg_color = "#f0f0f0"
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.video_path = tk.StringVar()
        self.sound_characteristic = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.model_loaded = False
        self.model = None
        self.processor = None
        
        self.model_var = tk.StringVar()
        self.prompt_source = tk.StringVar(value="preset")
        self.preset_var = tk.StringVar()
        
        # UI Setup
        self.create_widgets()
        
        # Check environment
        self.check_environment()

    def create_widgets(self):
        # 1. Top Section (Load Button + Title)
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        # Load Model Button (Top Left)
        self.load_btn = tk.Button(top_frame, text="Load Model", command=self.start_loading_thread,
                                  font=("Helvetica", 10, "bold"), bg="#2196F3", fg="white")
        self.load_btn.pack(side="left")
        
        # Model Dropdown (Next to Load Button)
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=25)
        self.model_combo.pack(side="left", padx=10)
        
        # Scan for models
        self.scan_models()
        
        # Title (Centered-ish)
        header_label = tk.Label(top_frame, text="SAM-AUDIO Sound Isolator", 
                                font=("Segeo UI", 16, "bold"), bg=self.bg_color)
        header_label.pack(side="left", expand=True, padx=(0, 200)) # Offset for balance

    def scan_models(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        
        models = []
        if os.path.exists(checkpoints_dir):
            for name in os.listdir(checkpoints_dir):
                full_path = os.path.join(checkpoints_dir, name)
                if os.path.isdir(full_path):
                    # Basic check for config.json to confirm it's likely a model
                    if os.path.exists(os.path.join(full_path, "config.json")):
                        models.append(name)
        
        if models:
            self.model_combo['values'] = models
            self.model_combo.current(0)
            self.load_btn.config(state="normal")
        else:
            self.model_combo['values'] = ("No models found!",)
            self.model_combo.current(0)
            self.load_btn.config(state="disabled")

        # 2. File Selection Section
        file_frame = tk.LabelFrame(self.root, text="Input Video", font=("Segoe UI", 10, "bold"), bg=self.bg_color, padx=10, pady=10)
        file_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(file_frame, text="File Path:", bg=self.bg_color).pack(side="left")
        entry_path = tk.Entry(file_frame, textvariable=self.video_path, width=40)
        entry_path.pack(side="left", padx=10, expand=True, fill="x")
        tk.Button(file_frame, text="Browse", command=self.select_video).pack(side="left")

        # 3. Sound Selection Section
        opts_frame = tk.LabelFrame(self.root, text="Sound to Isolate", font=("Segoe UI", 10, "bold"), bg=self.bg_color, padx=10, pady=10)
        opts_frame.pack(fill="x", padx=10, pady=5)
        
        # Grid layout for options
        # Row 0: Radio Buttons
        rb_preset = tk.Radiobutton(opts_frame, text="Use Preset", variable=self.prompt_source, 
                                   value="preset", command=self.toggle_inputs, bg=self.bg_color)
        rb_preset.grid(row=0, column=0, sticky="w")
        
        rb_custom = tk.Radiobutton(opts_frame, text="Custom Text", variable=self.prompt_source, 
                                   value="custom", command=self.toggle_inputs, bg=self.bg_color)
        rb_custom.grid(row=1, column=0, sticky="w")
        
        # Row 0: Preset Dropdown
        self.combo_presets = ttk.Combobox(opts_frame, textvariable=self.preset_var, state="readonly")
        self.combo_presets['values'] = ("Vocals", "Music", "Man", "Woman", "Static", "Nature", "Animals", "Wind", "Traffic")
        self.combo_presets.current(0) # Default to 'Vocals'
        self.combo_presets.grid(row=0, column=1, padx=10, sticky="ew")
        
        # Row 1: Custom Entry
        self.entry_custom = tk.Entry(opts_frame, textvariable=self.sound_characteristic, width=30)
        self.entry_custom.grid(row=1, column=1, padx=10, sticky="ew")
        
        opts_frame.columnconfigure(1, weight=1)
        
        # Initial State
        self.toggle_inputs()

        # 4. Action Section
        self.process_btn = tk.Button(self.root, text="ISOLATE SOUND", command=self.start_processing_thread, 
                                     font=("Segoe UI", 12, "bold"), bg="#4CAF50", fg="white", 
                                     state="disabled", height=2, width=20)
        self.process_btn.pack(pady=20)

        # 5. Feedback Section (Bottom)
        # Status Label (Very Bottom)
        status_label = tk.Label(self.root, textvariable=self.status_var, bg="#e0e0e0", relief="sunken", anchor="w")
        status_label.pack(side="bottom", fill="x")

        # Progress Bar (Above Status)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(side="bottom", pady=(0, 5), padx=10, fill="x")

    def toggle_inputs(self):
        choice = self.prompt_source.get()
        if choice == "preset":
            self.combo_presets.config(state="readonly")
            self.entry_custom.config(state="disabled", bg="#f0f0f0")
        else:
            self.combo_presets.config(state="disabled")
            self.entry_custom.config(state="normal", bg="white")

    def check_environment(self):
        # Check if ffmpeg is in path
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if SAM_AVAILABLE:
                self.process_btn.config(state="normal")
            else:
                self.status_var.set("Error: SAM-AUDIO library not found.")
                messagebox.showerror("Error", "SAM-AUDIO library could not be imported. Check dependencies.")
        except FileNotFoundError:
            self.status_var.set("Error: FFMPEG not found in PATH.")
            messagebox.showerror("Error", "FFMPEG is not installed or not in PATH.")

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path.set(file_path)

    def start_loading_thread(self):
        if self.model_loaded:
            messagebox.showinfo("Info", "Model is already loaded.")
            return
        
        self.load_btn.config(state="disabled")
        self.process_btn.config(state="disabled")
        thread = threading.Thread(target=self.load_model)
        thread.daemon = True
        thread.start()

    def load_model(self):
        try:
            # 1. Setup
            selected_model_name = self.model_var.get()
            if selected_model_name == "No models found!":
                 messagebox.showerror("Error", "No valid model selected.")
                 return

            self.update_status(f"Loading {selected_model_name}...")
            self.update_progress(0, "determinate")
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(base_dir, "checkpoints", selected_model_name)
            
            model_id = checkpoint_path
            is_local = True # Always local with new logic
            
            print(f"Loading model from: {model_id}")

            # 2. Load Config
            self.update_status("Loading configuration...")
            self.update_progress(10)
            
            if is_local:
                config_path = os.path.join(model_id, "config.json")
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = SAMAudioConfig(**config_dict)
            else:
                # Fallback to standard loading for remote models (simpler)
                self.update_progress(10, "indeterminate")
                self.progress.start(10)
                self.model = SAMAudio.from_pretrained(model_id)
                self.processor = SAMAudioProcessor.from_pretrained(model_id)
                self.progress.stop()
                # Skip to end
                config = None 

            if config:
                # 3. Instantiate Model
                self.update_status("Initializing model structure...")
                self.update_progress(20)
                self.model = SAMAudio(config)
                
                # 4. Load Processor (Fast)
                self.processor = SAMAudioProcessor.from_pretrained(model_id)

                # 5. Load Weights (Slowest Part)
                self.update_status("Reading weights from disk... (Large File)")
                self.update_progress(30)
                
                weights_path = os.path.join(model_id, "checkpoint.pt")
                # map_location='cpu' first to avoid OOM or CUDA context spam before ready? 
                # Actually loading directly to CPU is standard.
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                
                # 6. Apply Weights
                self.update_status("Applying weights to model...")
                self.update_progress(70)
                self.model.load_state_dict(state_dict, strict=True)
            
            # 7. Move to Device
            if torch.cuda.is_available():
                self.update_status("Moving model to CUDA...")
                self.update_progress(80)
                self.model = self.model.eval().cuda()
                self.update_status("Model loaded on CUDA.")
            else:
                self.model = self.model.eval()
                self.update_status("Model loaded on CPU.")
            
            self.model_loaded = True
            self.update_progress(100)
            
            # Enable process button
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.load_btn.config(state="normal", text="Model Loaded", bg="#8BC34A"))

        except Exception as e:
            self.progress.stop()
            self.update_progress(0, "determinate")
            self.update_status("Error loading model.")
            print(e)
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

    def start_processing_thread(self):
        if not self.video_path.get():
            messagebox.showwarning("Warning", "Please select a video file.")
            return
        
        # Determine actual prompt
        if self.prompt_source.get() == "preset":
            prompt = self.preset_var.get()
            if not prompt:
                messagebox.showwarning("Warning", "Please select a preset sound.")
                return
        else:
            prompt = self.sound_characteristic.get()
            if not prompt:
                messagebox.showwarning("Warning", "Please enter a custom sound description.")
                return

        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first.")
            return

        self.process_btn.config(state="disabled")
        # Pass prompt explicitly or set it to the variable used by run_process
        self.sound_characteristic.set(prompt) # Ensure the variable holds the correct value
        
        thread = threading.Thread(target=self.run_process)
        thread.daemon = True
        thread.start()

    def run_process(self):
        try:
            video_file = self.video_path.get()
            prompt = self.sound_characteristic.get()
            base_dir = os.path.dirname(video_file)
            filename = os.path.basename(video_file)
            name, ext = os.path.splitext(filename)
            
            # Temporary audio file
            temp_audio = os.path.join(base_dir, "temp_extracted_audio.wav")
            
            # Defines paths for output
            target_wav = os.path.join(base_dir, "temp_target.wav")
            residual_wav = os.path.join(base_dir, "temp_residual.wav")
            out_target_video = os.path.join(base_dir, f"{name}_isolated{ext}")
            out_residual_video = os.path.join(base_dir, f"{name}_residual{ext}")

            # Reset Progress
            self.update_progress(0, "determinate")

            # 1. Extract Audio
            self.update_status("Extracting audio from video...")
            self.update_progress(10)
            subprocess.run([
                "ffmpeg", "-y", "-i", video_file, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                temp_audio
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 2. Chunking and Inference
            self.update_status(f"Isolating '{prompt}' (processing chunks)...")
            
            # Load full audio
            waveform, sr = torchaudio.load(temp_audio)
            # waveform is (C, T), C=1
            
            seconds_per_chunk = 30
            samples_per_chunk = seconds_per_chunk * sr
            total_samples = waveform.shape[1]
            num_chunks = math.ceil(total_samples / samples_per_chunk)
            
            target_chunks = []
            residual_chunks = []
            
            for i in range(num_chunks):
                start = i * samples_per_chunk
                end = min((i + 1) * samples_per_chunk, total_samples)
                
                chunk_waveform = waveform[:, start:end]
                
                # Check for silence or very short chunks? SAMAudio might handle it.
                if chunk_waveform.shape[1] < 100: 
                    # If tiny chunk at end, just pad or skip? 
                    # Skipped for simplicity or pad with zeros
                    continue

                # Prepare inputs
                # processor handles tensors: List[Tensor] -> [batch, T]
                # It expects tensors to be already at the correct sample rate and mono/stereo as needed.
                target_sr = self.processor.audio_sampling_rate
                
                if sr != target_sr:
                     chunk_input = torchaudio.functional.resample(chunk_waveform, sr, target_sr)
                else:
                     chunk_input = chunk_waveform

                batch = self.processor(
                    audios=[chunk_input],
                    descriptions=[prompt]
                )
                
                if torch.cuda.is_available():
                    batch = batch.to("cuda")
                
                with torch.inference_mode():
                    result = self.model.separate(batch, predict_spans=True)
                
                # Collect results
                # result.target[0] is 1D tensor usually
                t_tens = result.target[0].cpu()
                r_tens = result.residual[0].cpu()
                
                if t_tens.ndim == 1: t_tens = t_tens.unsqueeze(0)
                if r_tens.ndim == 1: r_tens = r_tens.unsqueeze(0)
                
                # Resample BACK to original SR for saving? 
                # Or just save at model SR. Saving at model SR is better quality preservation if upsampled.
                # However, we merge chunks. If we change SR, the total length in samples changes.
                # We calculated num_chunks based on original SR.
                # If we concatenate resampled chunks, the final audio will be at target_sr.
                # We need to update the save variable 'sample_rate' to 'target_sr'.
                
                target_chunks.append(t_tens)
                residual_chunks.append(r_tens)
                
                # Update progress
                # Progress ranges from 10 to 90
                percent = 10 + (80 * (i + 1) / num_chunks)
                self.update_progress(percent)
                self.update_status(f"Processed chunk {i+1}/{num_chunks}")

            # Concatenate
            full_target = torch.cat(target_chunks, dim=1)
            full_residual = torch.cat(residual_chunks, dim=1)

            # 3. Save Audio
            self.update_status("Saving merged audio...")
            self.update_progress(90)
            
            # Since we resampled chunks to target_sr, the output is at target_sr
            save_sr = self.processor.audio_sampling_rate
            
            torchaudio.save(target_wav, full_target, save_sr)
            torchaudio.save(residual_wav, full_residual, save_sr)

            # 5. Remux Video
            self.update_status("Creating output videos...")
            self.update_progress(85)
            
            # Mux Target
            subprocess.run([
                "ffmpeg", "-y", "-i", video_file, "-i", target_wav,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                out_target_video
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Mux Residual
            subprocess.run([
                "ffmpeg", "-y", "-i", video_file, "-i", residual_wav,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                out_residual_video
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Cleanup
            if os.path.exists(temp_audio): os.remove(temp_audio)
            if os.path.exists(target_wav): os.remove(target_wav)
            if os.path.exists(residual_wav): os.remove(residual_wav)
            
            self.update_progress(100)
            self.update_status("Processing Complete!")
            
            messagebox.showinfo("Success", f"Created:\n{out_target_video}\n{out_residual_video}")

        except Exception as e:
            self.progress.stop() # Ensure animation stops on error
            self.update_progress(0, "determinate")
            self.update_status("Error occurred.")
            print(e)
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.process_btn.config(state="normal")
            
    def update_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def update_progress(self, value, mode=None):
        if mode:
            self.progress.config(mode=mode)
        self.progress['value'] = value
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundIsolatorApp(root)
    root.mainloop()
