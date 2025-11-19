import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import json
import glob
import urllib.request  
import zipfile         
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Try absolute import (runtime), fallback to relative import for editors/type-checkers
try:
    from feature_extractors.csv_extract_eGeMAPS_FUNCTION import extract_egemaps_features, egemaps_features_v2
    from feature_extractors.extract_features_custom import extract_custom_features
except Exception:
    from ..feature_extractors.csv_extract_eGeMAPS_FUNCTION import extract_egemaps_features, egemaps_features_v2
    from ..feature_extractors.extract_features_custom import extract_custom_features


def download_opensmile(self):
    """Download and install openSMILE automatically"""
    import urllib.request
    import zipfile
    import threading
    
    def download_thread():
        try:
            self.set_status("Downloading openSMILE (~50MB)... Please wait", "progress")
            self.progress.start(10)
            self.extract_button.config(state='disabled')
            
            # Percorso di download - cartella temporanea dell'utente
            download_folder = os.path.join(os.path.expanduser('~'), 'opensmile')
            os.makedirs(download_folder, exist_ok=True)
            
            # URL openSMILE 3.0.2 per Windows
            url = "https://github.com/audeering/opensmile/releases/download/v3.0.2/opensmile-3.0.2-win-x64.zip"
            zip_path = os.path.join(download_folder, 'opensmile.zip')
            
            # Download con progress callback
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) // total_size)
                    self.set_status(f"Downloading openSMILE... {percent}%", "progress")
            
            urllib.request.urlretrieve(url, zip_path, show_progress)
            
            self.set_status("Extracting openSMILE... Almost done", "progress")
            self.root.update()
            
            # Estrai nella stessa cartella
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_folder)
            
            # Rimuovi il file zip
            os.remove(zip_path)
            
            # Cerca la cartella estratta (potrebbe essere opensmile-3.0.2-win-x64)
            extracted_folders = [f for f in os.listdir(download_folder) 
                               if os.path.isdir(os.path.join(download_folder, f)) and 'opensmile' in f.lower()]
            
            if extracted_folders:
                # Rinomina la cartella estratta in 'opensmile' se necessario
                extracted_path = os.path.join(download_folder, extracted_folders[0])
                final_path = os.path.join(os.path.expanduser('~'), 'opensmile')
                
                if extracted_path != final_path:
                    # Se esiste già la cartella finale, rimuovila
                    import shutil
                    if os.path.exists(final_path) and final_path != extracted_path:
                        try:
                            shutil.rmtree(final_path)
                        except:
                            pass
                    
                    # Sposta/rinomina
                    try:
                        shutil.move(extracted_path, final_path)
                    except:
                        final_path = extracted_path
            
            # Prova a rilevare i path
            detected = auto_detect_opensmile()
            
            if detected:
                self.save_config(detected)
                self.set_status("✓ openSMILE installed successfully!", "success")
                messagebox.showinfo(
                    "Installation Complete",
                    f"openSMILE has been installed to:\n{download_folder}\n\n"
                    "You can now start extracting features!"
                )
            else:
                self.set_status("Installation completed but configuration failed", "error")
                messagebox.showerror(
                    "Configuration Error",
                    f"openSMILE was downloaded to:\n{download_folder}\n\n"
                    "But automatic configuration failed.\n"
                    "Please configure paths manually in Settings."
                )
            
        except Exception as e:
            self.set_status(f"Download failed: {str(e)}", "error")
            messagebox.showerror(
                "Download Failed",
                f"Failed to download openSMILE:\n{str(e)}\n\n"
                "Please:\n"
                "1. Check your internet connection\n"
                "2. Download manually from:\n"
                "   github.com/audeering/opensmile/releases\n"
                "3. Configure paths in Settings"
            )
        finally:
            self.progress.stop()
            self.extract_button.config(state='normal')
            self.root.update()
    
    # Esegui in thread separato per non bloccare la GUI
    thread = threading.Thread(target=download_thread, daemon=True)
    thread.start()



def auto_detect_opensmile():
    """Cerca openSMILE in posizioni comuni di installazione"""
    possible_locations = [
        # Cartelle comuni Windows
        r'C:\opensmile',
        r'C:\Program Files\opensmile',
        r'C:\Program Files (x86)\opensmile',
        
        # User folder
        os.path.join(os.path.expanduser('~'), 'opensmile'),
        
        # Desktop
        os.path.join(os.path.expanduser('~'), 'Desktop', 'opensmile'),
        
        # Documents
        os.path.join(os.path.expanduser('~'), 'Documents', 'opensmile'),
        
        # Stesso livello del progetto
        os.path.join(os.path.dirname(__file__), '..', '..', 'opensmile'),
        
        # Livello superiore
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'opensmile')),
    ]
    
    for base_path in possible_locations:
        if not os.path.exists(base_path):
            continue
        
        # Percorsi standard openSMILE
        smile_exe = os.path.join(base_path, 'build', 'progsrc', 'smilextract', 'Release', 'SMILExtract.exe')
        compare_conf = os.path.join(base_path, 'config', 'compare16', 'ComParE_2016.conf')
        egemaps_conf = os.path.join(base_path, 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf')
        
        # Verifica che tutti i file esistano
        if all(os.path.exists(p) for p in [smile_exe, compare_conf, egemaps_conf]):
            print(f"✓ openSMILE found at: {base_path}")
            return {
                'SMILE_path': os.path.normpath(smile_exe),
                'Compare2016_config_path': os.path.normpath(compare_conf),
                'eGeMAPS_config_path': os.path.normpath(egemaps_conf)
            }
    
    return None



class AudioFeatureExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Feature Extractor")
        self.root.geometry("800x650")
        self.root.resizable(True, True)
        self.root.minsize(700, 550)
        
        # Modern color scheme
        self.colors = {
            'bg': '#f5f6fa',
            'primary': '#3498db',
            'primary_dark': '#2980b9',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'dark': '#2c3e50',
            'light': '#ecf0f1',
            'white': '#ffffff',
            'border': '#bdc3c7'
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['bg'])
        
        # Config file path
        self.config_file = os.path.join(os.path.dirname(__file__), 'gui_config.json')
        
        # Variables
        self.selected_path = tk.StringVar()
        self.input_type = tk.StringVar(value="file")
        self.extract_egemaps = tk.BooleanVar(value=False)
        self.extract_custom = tk.BooleanVar(value=False)
        
        # AGGIUNTO: Traccia le feature selezionate
        self.selected_egemaps_features = []
        self.selected_custom_features = []
        
        # Load configuration from JSON
        self.config = self.load_config()
        
        # AGGIUNTO: Carica le liste di feature dai JSON
        self.load_feature_lists()
        
        # Configure styles
        self.configure_styles()
        
        self.create_widgets()
        
        # Check configuration after widgets are created
        self.check_initial_config()
        
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            'root_folder_path': '',
            'SMILE_path': '',
            'Compare2016_config_path': '',
            'eGeMAPS_config_path': ''
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge with default to ensure all keys exist
                    return {**default_config, **config}
            else:
                # Create config file with default values
                self.save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return default_config
    
    def save_config(self, updates):
        """Save configuration to JSON file"""
        try:
            # Update config in memory
            self.config.update(updates)
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

    # AGGIUNTO: Metodo per caricare le liste di feature
    def load_feature_lists(self):
        """Load feature lists from JSON files"""
        try:
            # Load custom features
            custom_json_path = os.path.join(os.path.dirname(__file__), 'custom_features_list.json')
            if os.path.exists(custom_json_path):
                with open(custom_json_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                    self.custom_features_list = custom_data.get('features', [])
                    # Default: all selected
                    self.selected_custom_features = self.custom_features_list.copy()
            else:
                self.custom_features_list = []
                self.selected_custom_features = []
                print(f"Warning: {custom_json_path} not found")
            
            # Load eGeMAPS features
            egemaps_json_path = os.path.join(os.path.dirname(__file__), 'eGeMAPS_features_list.json')
            if os.path.exists(egemaps_json_path):
                with open(egemaps_json_path, 'r', encoding='utf-8') as f:
                    egemaps_data = json.load(f)
                    self.egemaps_features_list = egemaps_data.get('features', [])
                    # Default: all selected
                    self.selected_egemaps_features = self.egemaps_features_list.copy()
            else:
                self.egemaps_features_list = []
                self.selected_egemaps_features = []
                print(f"Warning: {egemaps_json_path} not found")
                
        except Exception as e:
            print(f"Error loading feature lists: {e}")
            self.custom_features_list = []
            self.egemaps_features_list = []
            self.selected_custom_features = []
            self.selected_egemaps_features = []

    def check_initial_config(self):
        """Check configuration on startup and notify user"""
        # Verifica se i path sono già configurati
        if not self.config.get('SMILE_path') or not os.path.exists(self.config.get('SMILE_path', '')):
            # Prova auto-detect
            detected = auto_detect_opensmile()
            
            if detected:
                # Trovato! Salva automaticamente
                self.save_config(detected)
                self.set_status(f"openSMILE auto-detected and configured", "success")
                return
            else:
                # Non trovato, chiedi se scaricare
                response = messagebox.askyesnocancel(
                    "openSMILE Not Found",
                    "openSMILE was not found on your system.\n\n"
                    "Would you like to:\n"
                    "• YES - Download and install automatically (~50MB)\n"
                    "• NO - Select existing installation manually\n"
                    "• CANCEL - Configure later\n",
                    icon='question'
                )
                
                if response is True:  # YES - download
                    self.download_opensmile()
                elif response is False:  # NO - manual
                    self.open_opensmile_settings()
                else:  # CANCEL
                    self.set_status("openSMILE not configured - Click Settings when ready", "warning")
                return
        
        # Già configurato
        self.set_status("Ready - Select audio files and feature sets to begin", "success")
    
    def download_opensmile(self):
        """Wrapper to call the module-level function"""
        download_opensmile(self)

    def configure_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        
        # Configure frame styles
        style.configure('Card.TFrame', background=self.colors['white'], relief='flat')
        style.configure('Main.TFrame', background=self.colors['bg'])
        
        # Configure label styles
        style.configure('Title.TLabel', 
                       background=self.colors['white'],
                       foreground=self.colors['dark'],
                       font=('Segoe UI', 11, 'bold'))
        
        style.configure('Header.TLabel',
                       background=self.colors['primary'],
                       foreground=self.colors['white'],
                       font=('Segoe UI', 16, 'bold'),
                       padding=20)
        
        style.configure('Status.TLabel',
                       background=self.colors['light'],
                       foreground=self.colors['dark'],
                       font=('Segoe UI', 9),
                       padding=10)
        
        # Configure button styles
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        
        style.configure('Settings.TButton',
                       font=('Segoe UI', 9),
                       padding=8)
        
        # Configure checkbutton styles
        style.configure('Feature.TCheckbutton',
                       background=self.colors['white'],
                       font=('Segoe UI', 10),
                       padding=8)


    def create_widgets(self):
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # Main container expandable
        self.root.rowconfigure(0, weight=0)  # Header fixed
        self.root.rowconfigure(2, weight=0)  # Status bar fixed
        
        # Header with title
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=70)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🎵 Audio Feature Extractor",
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Segoe UI', 18, 'bold')
        )
        title_label.pack(side='left', padx=30, pady=15)
        
        # Settings button in header
        settings_btn = tk.Button(
            header_frame,
            text="⚙ Settings",
            bg=self.colors['primary_dark'],
            fg=self.colors['white'],
            font=('Segoe UI', 9, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self.open_opensmile_settings,
            activebackground=self.colors['dark'],
            activeforeground=self.colors['white']
        )
        settings_btn.pack(side='right', padx=30)
        self.create_tooltip(settings_btn, "Configure openSMILE paths")


        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)
        main_container.columnconfigure(0, weight=1)


        # Input Selection Card
        input_card = self.create_card(main_container, "📁 Input Selection")
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        # Radio buttons with modern look
        radio_frame = tk.Frame(input_card, bg=self.colors['white'])
        radio_frame.pack(fill='x', padx=20, pady=(10, 15))
        
        rb1 = tk.Radiobutton(
            radio_frame,
            text="📄 Single Audio File",
            variable=self.input_type,
            value="file",
            bg=self.colors['white'],
            fg=self.colors['dark'],
            font=('Segoe UI', 10),
            selectcolor=self.colors['light'],
            activebackground=self.colors['white'],
            cursor='hand2'
        )
        rb1.pack(side='left', padx=(0, 30))
        
        rb2 = tk.Radiobutton(
            radio_frame,
            text="📂 Folder",
            variable=self.input_type,
            value="folder",
            bg=self.colors['white'],
            fg=self.colors['dark'],
            font=('Segoe UI', 10),
            selectcolor=self.colors['light'],
            activebackground=self.colors['white'],
            cursor='hand2'
        )
        rb2.pack(side='left')
        
        # Path selection with modern entry
        path_frame = tk.Frame(input_card, bg=self.colors['white'])
        path_frame.pack(fill='x', padx=20, pady=(0, 20))
        path_frame.columnconfigure(0, weight=1)
        
        entry = tk.Entry(
            path_frame,
            textvariable=self.selected_path,
            font=('Segoe UI', 10),
            relief='solid',
            borderwidth=1,
            bg=self.colors['light'],
            fg=self.colors['dark']
        )
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 15), ipady=8)
        
        browse_btn = tk.Button(
            path_frame,
            text="Browse",
            command=self.browse_path,
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=25,
            pady=8,
            cursor='hand2',
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['white']
        )
        browse_btn.grid(row=0, column=1)

        # Feature Selection Card - MODIFICATO
        feature_card = self.create_card(main_container, "🎛️ Feature Selection")
        feature_card.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        features_inner = tk.Frame(feature_card, bg=self.colors['white'])
        features_inner.pack(fill='x', padx=20, pady=(10, 20))
        
        # eGeMAPS checkbox with configure button
        cb1_frame = tk.Frame(features_inner, bg=self.colors['white'])
        cb1_frame.pack(fill='x', pady=8)
        
        cb1 = tk.Checkbutton(
            cb1_frame,
            text="eGeMAPS Features",
            variable=self.extract_egemaps,
            bg=self.colors['white'],
            fg=self.colors['dark'],
            font=('Segoe UI', 11, 'bold'),
            selectcolor=self.colors['light'],
            activebackground=self.colors['white'],
            cursor='hand2'
        )
        cb1.pack(side='left')
        
        desc1 = tk.Label(
            cb1_frame,
            text="Extended Geneva Minimalistic Acoustic Parameter Set",
            bg=self.colors['white'],
            fg='#7f8c8d',
            font=('Segoe UI', 9)
        )
        desc1.pack(side='left', padx=(10, 10))
        
        # AGGIUNTO: Configure button per eGeMAPS
        config_btn_egemaps = tk.Button(
            cb1_frame,
            text="⚙ Configure",
            command=lambda: self.open_feature_selector('egemaps'),
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Segoe UI', 8, 'bold'),
            relief='flat',
            padx=12,
            pady=4,
            cursor='hand2',
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['white']
        )
        config_btn_egemaps.pack(side='right', padx=(0, 5))
        
        # Custom features checkbox with configure button
        cb2_frame = tk.Frame(features_inner, bg=self.colors['white'])
        cb2_frame.pack(fill='x', pady=8)
        
        cb2 = tk.Checkbutton(
            cb2_frame,
            text="Custom Features",
            variable=self.extract_custom,
            bg=self.colors['white'],
            fg=self.colors['dark'],
            font=('Segoe UI', 11, 'bold'),
            selectcolor=self.colors['light'],
            activebackground=self.colors['white'],
            cursor='hand2'
        )
        cb2.pack(side='left')
        
        desc2 = tk.Label(
            cb2_frame,
            text="Pause analysis, voicing metrics, and prosody features",
            bg=self.colors['white'],
            fg='#7f8c8d',
            font=('Segoe UI', 9)
        )
        desc2.pack(side='left', padx=(10, 10))
        
        # AGGIUNTO: Configure button per custom
        config_btn_custom = tk.Button(
            cb2_frame,
            text="⚙ Configure",
            command=lambda: self.open_feature_selector('custom'),
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Segoe UI', 8, 'bold'),
            relief='flat',
            padx=12,
            pady=4,
            cursor='hand2',
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['white']
        )
        config_btn_custom.pack(side='right', padx=(0, 5))

        # Action area
        action_frame = tk.Frame(main_container, bg=self.colors['bg'])
        action_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        # Center the content in action frame
        action_content = tk.Frame(action_frame, bg=self.colors['bg'])
        action_content.pack(expand=True)
        
        # Progress bar with modern style
        self.progress = ttk.Progressbar(
            action_content,
            mode='indeterminate',
            length=500,
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress.pack(pady=(0, 15))
        
        # Configure progress bar style
        style = ttk.Style()
        style.configure('Custom.Horizontal.TProgressbar',
                       background=self.colors['primary'],
                       troughcolor=self.colors['light'],
                       borderwidth=0,
                       thickness=8)
        
        # Large extract button
        self.extract_button = tk.Button(
            action_content,
            text="🚀 Extract Features",
            command=self.extract_features,
            bg=self.colors['success'],
            fg=self.colors['white'],
            font=('Segoe UI', 12, 'bold'),
            relief='flat',
            padx=50,
            pady=15,
            cursor='hand2',
            activebackground='#229954',
            activeforeground=self.colors['white']
        )
        self.extract_button.pack()
        
        # Keyboard shortcut label
        shortcut_label = tk.Label(
            action_content,
            text="Shortcut: Ctrl+E",
            bg=self.colors['bg'],
            fg='#95a5a6',
            font=('Segoe UI', 8)
        )
        shortcut_label.pack(pady=(8, 0))
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-e>', lambda e: self.extract_features())
        self.root.bind('<Control-E>', lambda e: self.extract_features())


        # Status bar
        status_frame = tk.Frame(self.root, bg=self.colors['light'], height=45)
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready - Select audio files and feature sets to begin",
            bg=self.colors['light'],
            fg=self.colors['dark'],
            font=('Segoe UI', 9),
            anchor='w'
        )
        self.status_label.pack(fill='both', padx=20, pady=12)


    def create_card(self, parent, title):
        """Create a modern card-style frame with title"""
        card = tk.Frame(parent, bg=self.colors['white'], relief='flat', borderwidth=1)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        title_bar = tk.Frame(card, bg=self.colors['white'])
        title_bar.pack(fill='x', padx=20, pady=(15, 5))
        
        title_label = tk.Label(
            title_bar,
            text=title,
            bg=self.colors['white'],
            fg=self.colors['dark'],
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack(side='left')
        
        # Separator line
        separator = tk.Frame(card, bg=self.colors['border'], height=1)
        separator.pack(fill='x', padx=20, pady=(10, 0))
        
        return card


    def set_status(self, message, status_type="info"):
        """Update status bar with message and icon"""
        icons = {
            'success': '✓',
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ',
            'progress': '⟳'
        }
        colors = {
            'success': self.colors['success'],
            'error': self.colors['danger'],
            'warning': self.colors['warning'],
            'info': self.colors['primary'],
            'progress': '#7f8c8d'
        }
        
        icon = icons.get(status_type, 'ℹ')
        color = colors.get(status_type, self.colors['dark'])
        
        self.status_label.config(
            text=f"{icon} {message}",
            fg=color
        )
        self.root.update()


    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                tooltip,
                text=text,
                background='#34495e',
                foreground='white',
                relief='flat',
                borderwidth=0,
                font=('Segoe UI', 9),
                padx=10,
                pady=6
            )
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def browse_path(self):
        """Open file or folder dialog based on input type"""
        if self.input_type.get() == "folder":
            path = filedialog.askdirectory(title="Select folder containing audio files")
        else:
            path = filedialog.askopenfilename(
                title="Select an audio file",
                filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
            )
        
        if path:
            path = path.replace('/', '\\')   
            self.selected_path.set(path)
            # Save only to JSON
            self.save_config({'root_folder_path': path})
            self.set_status(f"Selected: {os.path.basename(path)}", "success")

    # MODIFICATO: Metodo per aprire la finestra di selezione feature con radio buttons per eGeMAPS
    def open_feature_selector(self, feature_type):
        """Open a popup window to select specific features"""
        if feature_type == 'egemaps':
            title = "Configure eGeMAPS Features"
            available_features = self.egemaps_features_list
            current_selection = self.selected_egemaps_features if isinstance(self.selected_egemaps_features, list) else []
        else:  # custom
            title = "Select Custom Features"
            available_features = self.custom_features_list
            current_selection = self.selected_custom_features
        
        if not available_features and feature_type != 'egemaps':
            messagebox.showwarning("No Features", f"No {feature_type} features found in JSON file")
            return
        
        # Create popup window
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("650x650")
        dialog.configure(bg=self.colors['bg'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg=self.colors['primary'], height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text=f"📋 {title}",
            bg=self.colors['primary'],
            fg=self.colors['white'],
            font=('Segoe UI', 14, 'bold')
        )
        header_label.pack(pady=15)
        
        # AGGIUNTO: Radio buttons per eGeMAPS
        if feature_type == 'egemaps':
            # Frame per i radio button
            radio_frame = tk.Frame(dialog, bg=self.colors['white'])
            radio_frame.pack(fill='x', padx=20, pady=(20, 10))
            
            # Variabile locale per il dialog
            egemaps_selection_mode = tk.StringVar(value="custom")  # "custom", "all", "v2_all"
            
            tk.Label(
                radio_frame,
                text="Selection Mode:",
                bg=self.colors['white'],
                fg=self.colors['dark'],
                font=('Segoe UI', 10, 'bold')
            ).pack(anchor='w', pady=(0, 8))
            
            rb_custom = tk.Radiobutton(
                radio_frame,
                text="🎯 Custom selection (choose below)",
                variable=egemaps_selection_mode,
                value="custom",
                bg=self.colors['white'],
                fg=self.colors['dark'],
                font=('Segoe UI', 9),
                selectcolor=self.colors['light'],
                activebackground=self.colors['white'],
                cursor='hand2'
            )
            rb_custom.pack(anchor='w', pady=2)
            
            rb_v2_all = tk.Radiobutton(
                radio_frame,
                text="📊 All v2 standard (48 features)",
                variable=egemaps_selection_mode,
                value="v2_all",
                bg=self.colors['white'],
                fg=self.colors['dark'],
                font=('Segoe UI', 9),
                selectcolor=self.colors['light'],
                activebackground=self.colors['white'],
                cursor='hand2'
            )
            rb_v2_all.pack(anchor='w', pady=2)
            
            rb_all = tk.Radiobutton(
                radio_frame,
                text="🔥 All features (88 features)",
                variable=egemaps_selection_mode,
                value="all",
                bg=self.colors['white'],
                fg=self.colors['dark'],
                font=('Segoe UI', 9),
                selectcolor=self.colors['light'],
                activebackground=self.colors['white'],
                cursor='hand2'
            )
            rb_all.pack(anchor='w', pady=2)
            
            # Separatore
            ttk.Separator(dialog, orient='horizontal').pack(fill='x', padx=20, pady=10)
        
        # Info label
        info_frame = tk.Frame(dialog, bg=self.colors['bg'])
        info_frame.pack(fill='x', padx=20, pady=(10 if feature_type == 'egemaps' else 15, 10))
        
        info_text = "Select individual features below:" if feature_type == 'egemaps' else f"Select the features you want to extract ({len(available_features)} available)"
        info_label = tk.Label(
            info_frame,
            text=info_text,
            bg=self.colors['bg'],
            fg=self.colors['dark'],
            font=('Segoe UI', 9)
        )
        info_label.pack(anchor='w')
        
        # Select/Deselect all frame (solo per custom o mode custom per egemaps)
        select_frame = tk.Frame(dialog, bg=self.colors['bg'])
        select_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        select_all_var = tk.BooleanVar(value=True)
        
        feature_vars = {}
        checkboxes = []  # Lista per tenere traccia delle checkbox (solo eGeMAPS)
        
        def toggle_all():
            state = select_all_var.get()
            for var in feature_vars.values():
                var.set(state)
        
        select_all_cb = tk.Checkbutton(
            select_frame,
            text="Select / Deselect All",
            variable=select_all_var,
            command=toggle_all,
            bg=self.colors['bg'],
            fg=self.colors['dark'],
            font=('Segoe UI', 10, 'bold'),
            selectcolor=self.colors['light'],
            activebackground=self.colors['bg'],
            cursor='hand2'
        )
        select_all_cb.pack(anchor='w')
        
        # Scrollable frame for checkboxes
        canvas_frame = tk.Frame(dialog, bg=self.colors['white'])
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        canvas = tk.Canvas(canvas_frame, bg=self.colors['white'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Abilita lo scroll con la rotellina del mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)

        # Create checkboxes for each feature
        for i, feature in enumerate(available_features):
            var = tk.BooleanVar(value=(feature in current_selection))
            feature_vars[feature] = var
            
            cb = tk.Checkbutton(
                scrollable_frame,
                text=feature,
                variable=var,
                bg=self.colors['white'],
                fg=self.colors['dark'],
                font=('Segoe UI', 9),
                selectcolor=self.colors['light'],
                activebackground=self.colors['white'],
                anchor='w',
                cursor='hand2'
            )
            cb.pack(fill='x', padx=15, pady=3)
            
            if feature_type == 'egemaps':
                checkboxes.append(cb)  # Salva riferimento
            
            # Alternate background for better readability
            if i % 2 == 0:
                cb.configure(bg='#f8f9fa')

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # AGGIUNTO: Funzione per abilitare/disabilitare checkbox
        def update_checkboxes_state():
            if feature_type == 'egemaps':
                mode = egemaps_selection_mode.get()
                state = 'normal' if mode == 'custom' else 'disabled'
                for cb in checkboxes:
                    cb.config(state=state)
                select_all_cb.config(state=state)
        
        # Bind radio buttons per aggiornare stato checkbox
        if feature_type == 'egemaps':
            rb_custom.config(command=update_checkboxes_state)
            rb_v2_all.config(command=update_checkboxes_state)
            rb_all.config(command=update_checkboxes_state)
        
        # Buttons frame
        button_frame = tk.Frame(dialog, bg=self.colors['bg'])
        button_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        def save_selection():
            if feature_type == 'egemaps':
                mode = egemaps_selection_mode.get()
                
                if mode == "all":
                    # Tutte le 88 feature
                    self.selected_egemaps_features = "all"
                    count_msg = "ALL (88)"
                elif mode == "v2_all":
                    # Tutte le 48 v2
                    self.selected_egemaps_features = egemaps_features_v2.copy()
                    count_msg = f"{len(egemaps_features_v2)}"
                else:
                    # Custom selection
                    self.selected_egemaps_features = [
                        f for f, var in feature_vars.items() if var.get()
                    ]
                    if not self.selected_egemaps_features:
                        messagebox.showwarning("No Selection", "Please select at least one feature")
                        return
                    count_msg = f"{len(self.selected_egemaps_features)}"
                
                self.set_status(f"Selected {count_msg} eGeMAPS features", "success")
                
            else:  # custom
                self.selected_custom_features = [
                    f for f, var in feature_vars.items() if var.get()
                ]
                if not self.selected_custom_features:
                    messagebox.showwarning("No Selection", "Please select at least one feature")
                    return
                self.set_status(f"Selected {len(self.selected_custom_features)} custom features", "success")
            
            dialog.destroy()
        
        def cancel_selection():
            dialog.destroy()
        
        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel_selection,
            bg='#95a5a6',
            fg=self.colors['white'],
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=30,
            pady=10,
            cursor='hand2',
            activebackground='#7f8c8d',
            activeforeground=self.colors['white']
        )
        cancel_btn.pack(side='left', padx=(0, 10))
        
        # Save button
        save_btn = tk.Button(
            button_frame,
            text="Save Selection",
            command=save_selection,
            bg=self.colors['success'],
            fg=self.colors['white'],
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=30,
            pady=10,
            cursor='hand2',
            activebackground='#229954',
            activeforeground=self.colors['white']
        )
        save_btn.pack(side='right')
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

    def cleanup_lld_files(self, path):
        """Delete all LLD files created during feature extraction"""
        try:
            # Attendi che i processi rilascino i file
            time.sleep(0.5)
            self.root.update()
            
            if os.path.isfile(path):
                base_dir = os.path.dirname(path)
                pattern = os.path.join(base_dir, '*LLD.csv')
            else:
                pattern = os.path.join(path, '**', '*LLD.csv')
            
            lld_files = glob.glob(pattern, recursive=True)
            deleted_count = 0
            failed_files = []
            
            for lld_file in lld_files:
                # Prova fino a 3 volte con delay
                for attempt in range(3):
                    try:
                        if os.path.exists(lld_file):
                            os.remove(lld_file)
                            deleted_count += 1
                            print(f"Deleted: {lld_file}")
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(0.3)
                        else:
                            failed_files.append(os.path.basename(lld_file))
                            print(f"Could not delete (permission denied): {lld_file}")
                    except Exception as e:
                        failed_files.append(os.path.basename(lld_file))
                        print(f"Error deleting {lld_file}: {e}")
                        break
            
            if failed_files:
                print(f"Warning: Could not delete {len(failed_files)} file(s): {', '.join(failed_files[:3])}")
                
        except Exception as e:
            print(f"Warning: Error during LLD cleanup: {e}")

    def extract_features(self):
        """Main extraction function with improved error handling"""
        path = self.selected_path.get()
        
        if not path:
            self.set_status("Please select an audio file or folder first", "error")
            return
            
        if not os.path.exists(path):
            self.set_status(f"Path does not exist: {path}", "error")
            return
            
        if not (self.extract_egemaps.get() or self.extract_custom.get()):
            self.set_status("Please select at least one feature type to extract", "error")
            return
        
        # Verify that openSMILE is configured
        if not self.config.get('SMILE_path') or not os.path.exists(self.config.get('SMILE_path', '')):
            self.set_status("openSMILE not configured - Click Settings first", "error")
            return


        # Disable button and start progress
        self.extract_button.config(state='disabled', bg='#95a5a6')
        self.progress.start(10)
        self.root.update()


        try:
            # eGeMAPS extraction
            if self.extract_egemaps.get():
                out_egemaps = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    initialfile="extracted_features_eGeMAPS.csv"
                )
                if not out_egemaps:
                    self.set_status("Operation cancelled by user", "warning")
                    return
                
                # Rimuovi file esistente se presente
                if os.path.exists(out_egemaps):
                    try:
                        os.remove(out_egemaps)
                        self.set_status(f"Removed existing file: {os.path.basename(out_egemaps)}", "info")
                    except Exception as e:
                        self.set_status(f"Warning: Could not remove existing file: {e}", "warning")
                    
                self.set_status("Extracting eGeMAPS features...", "progress")
                
                # MODIFICATO: Determina cosa passare allo script
                if self.selected_egemaps_features == "all":
                    features_to_extract = "all"
                elif isinstance(self.selected_egemaps_features, list):
                    features_to_extract = self.selected_egemaps_features
                else:
                    features_to_extract = None
                
                extract_egemaps_features(path, out_egemaps, selected_features=features_to_extract)
                self.set_status("eGeMAPS extraction completed successfully", "success")


            # Custom extraction
            if self.extract_custom.get():
                try:
                    try:
                        from feature_extractors.extract_feature_batch_LLD import generate_lld_for_file, generate_lld_in_tree
                    except Exception:
                        from ..feature_extractors.extract_feature_batch_LLD import generate_lld_for_file, generate_lld_in_tree
                    
                    if os.path.isfile(path) and path.lower().endswith('.wav'):
                        self.set_status("Generating LLD for file...", "progress")
                        generate_lld_for_file(path)
                    else:
                        self.set_status("Generating LLD files in folder...", "progress")
                        generate_lld_in_tree(path)
                        
                    self.set_status("LLD generation completed", "success")
                except Exception as e:
                    self.set_status(f"Error generating LLD files: {str(e)}", "error")
                    return


                out_custom = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    initialfile="extracted_features_custom.csv"
                )
                if not out_custom:
                    self.set_status("Operation cancelled by user", "warning")
                    return
                
                # Rimuovi file esistente se presente
                if os.path.exists(out_custom):
                    try:
                        os.remove(out_custom)
                        self.set_status(f"Removed existing file: {os.path.basename(out_custom)}", "info")
                    except Exception as e:
                        self.set_status(f"Warning: Could not remove existing file: {e}", "warning")
                    
                self.set_status("Extracting custom features...", "progress")
                extract_custom_features(path, output_path=out_custom, selected_features=self.selected_custom_features)
                self.set_status("Custom extraction completed successfully", "success")


            self.set_status("All feature extraction completed successfully!", "success")
            
        except Exception as e:
            self.set_status(f"Error during extraction: {str(e)}", "error")
        finally:
            # Assicurati che cleanup_lld_files sia chiamato
            try:
                self.cleanup_lld_files(path)
            except Exception as e:
                print(f"Warning: Error during LLD cleanup: {e}")
            
            # Re-enable button and stop progress
            self.extract_button.config(state='normal', bg=self.colors['success'])
            self.progress.stop()
            self.root.update()



    def open_opensmile_settings(self):
        """Open dialog to configure openSMILE paths"""
        folder = filedialog.askdirectory(title="Select openSMILE root folder")
        if not folder:
            self.set_status("openSMILE configuration cancelled", "warning")
            return


        # Expected relative locations
        smile_rel = os.path.join('build', 'progsrc', 'smilextract', 'Release', 'SMILExtract.exe')
        compare_rel = os.path.join('config', 'compare16', 'ComParE_2016.conf')
        egemaps_rel = os.path.join('config', 'egemaps', 'v02', 'eGeMAPSv02.conf')


        smile_path = os.path.abspath(os.path.join(folder, smile_rel))
        compare_path = os.path.abspath(os.path.join(folder, compare_rel))
        egemaps_path = os.path.abspath(os.path.join(folder, egemaps_rel))


        # Check for missing files
        missing = []
        if not os.path.exists(smile_path):
            missing.append("SMILExtract.exe")
        if not os.path.exists(compare_path):
            missing.append("ComParE_2016.conf")
        if not os.path.exists(egemaps_path):
            missing.append("eGeMAPSv02.conf")


        if missing:
            self.set_status(f"Missing openSMILE files: {', '.join(missing)}", "error")
            return


        # Normalize paths
        smile_path = os.path.normpath(smile_path)
        compare_path = os.path.normpath(compare_path)
        egemaps_path = os.path.normpath(egemaps_path)


        # Save only to config file
        self.save_config({
            'SMILE_path': smile_path,
            'Compare2016_config_path': compare_path,
            'eGeMAPS_config_path': egemaps_path
        })
        self.set_status("openSMILE paths configured successfully", "success")



def main():
    root = tk.Tk()
    app = AudioFeatureExtractorGUI(root)
    root.mainloop()



if __name__ == "__main__":
    main()
