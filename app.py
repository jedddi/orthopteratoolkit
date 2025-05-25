import streamlit as st
import os
from PIL import Image
import base64

# Import module functions
from modules.dataset_browser import dataset_browser_app
from modules.call_segmentation import call_segmentation_app
from modules.batch_processor import batch_processor_app
from modules.silence_detector import silence_detector_app
from modules.duration_normalizer import duration_normalizer_app

# Set page configuration
st.set_page_config(
    page_title="Orthoptera Analysis Toolkit",
    page_icon="🦗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and display logo
def add_logo():
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=200)
    else:
        # If no logo exists, display a text header
        st.sidebar.title("🦗 Orthoptera Analysis Suite")

# Main app
def main():
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Initialize core settings in session state
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = ""
    if 'output_path' not in st.session_state:
        st.session_state.output_path = ""
    if 'normalization' not in st.session_state:
        st.session_state.normalization = "minmax"
    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = "energy"
    if 'min_duration' not in st.session_state:
        st.session_state.min_duration = 0.05
    if 'min_silence' not in st.session_state:
        st.session_state.min_silence = 0.05
    if 'energy_threshold' not in st.session_state:
        st.session_state.energy_threshold = 0.15
    if 'peak_distance' not in st.session_state:
        st.session_state.peak_distance = 20
    if 'peak_prominence' not in st.session_state:
        st.session_state.peak_prominence = 0.1
    if 'peak_width' not in st.session_state:
        st.session_state.peak_width = 10
    
    # Add logo to sidebar
    add_logo()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "Home",
        "Dataset Browser",
        "Call Segmentation",
        "Batch Processing",
        "Silent Audio Detector",
        "Duration Normalizer",
        "Settings"
    ]
    
    selected_page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.current_page))
    st.session_state.current_page = selected_page
    
    # Show shared settings in sidebar if we're not on the Settings page
    if selected_page != "Settings":
        st.sidebar.title("Quick Settings")
        # Only show the most essential settings
        st.session_state.dataset_path = st.sidebar.text_input("Dataset Path", value=st.session_state.dataset_path)
        st.sidebar.caption("Enter the full path to your dataset folder")
        
        st.session_state.output_path = st.sidebar.text_input("Output Path", value=st.session_state.output_path)
        st.sidebar.caption("Enter the full path where processed files will be saved")
        
        # Show path helper
        with st.sidebar.expander("Need help with paths?"):
            st.write("""
            **Example paths:**
            - Windows: `C:\\Users\\YourName\\Documents\\Orthoptera_Dataset`
            - Mac/Linux: `/home/username/Orthoptera_Dataset`
            
            You can copy-paste the path from your file explorer/finder.
            """)

    # Render the selected page
    if selected_page == "Home":
        render_home_page()
    elif selected_page == "Dataset Browser":
        dataset_browser_app()
    elif selected_page == "Call Segmentation":
        call_segmentation_app()
    elif selected_page == "Batch Processing":
        batch_processor_app()
    elif selected_page == "Silent Audio Detector":
        silence_detector_app()
    elif selected_page == "Duration Normalizer":
        duration_normalizer_app()
    elif selected_page == "Settings":
        render_settings_page()

def render_home_page():
    st.title("🦗 Orthoptera Analysis Suite")
    st.write("Welcome to the Orthoptera Analysis Suite, an integrated tool for analyzing, processing, and managing orthoptera call recordings.")
    
    # Overview
    st.header("Overview")
    st.write("""
    This application provides a comprehensive set of tools for researchers and enthusiasts working with orthoptera audio recordings. 
    It integrates several key functionalities in one unified interface.
    """)
    
    # Feature cards using columns
    st.header("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Dataset Browser")
        st.write("""
        Browse your dataset by species, visualize waveforms and spectrograms, 
        and listen to original and normalized audio.
        """)
        if st.button("Open Dataset Browser", key="home_browser"):
            st.session_state.current_page = "Dataset Browser"
            st.rerun()
    
    with col2:
        st.subheader("✂️ Call Segmentation")
        st.write("""
        Segment individual calls from longer recordings using energy-based or 
        peak detection methods. Preview and validate detected segments.
        """)
        if st.button("Open Call Segmentation", key="home_segmentation"):
            st.session_state.current_page = "Call Segmentation"
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔄 Batch Processing")
        st.write("""
        Process multiple files at once to extract call segments 
        across your entire dataset or selected species.
        """)
        if st.button("Open Batch Processing", key="home_batch"):
            st.session_state.current_page = "Batch Processing"
            st.rerun()
    
    with col4:
        st.subheader("🔇 Silent Audio Detector")
        st.write("""
        Scan your dataset for silent or nearly-silent audio files 
        that might need to be removed or fixed.
        """)
        if st.button("Open Silent Audio Detector", key="home_silence"):
            st.session_state.current_page = "Silent Audio Detector"
            st.rerun()
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("⏱️ Duration Normalizer")
        st.write("""
        Standardize the duration of audio files by adding 
        silence padding to create a uniform dataset.
        """)
        if st.button("Open Duration Normalizer", key="home_duration"):
            st.session_state.current_page = "Duration Normalizer"
            st.rerun()
    
    with col6:
        st.subheader("⚙️ Settings")
        st.write("""
        Configure application-wide settings for audio processing, 
        detection algorithms, and file handling.
        """)
        if st.button("Open Settings", key="home_settings"):
            st.session_state.current_page = "Settings"
            st.rerun()
    
    # Getting started
    st.header("Getting Started")
    st.write("""
    1. Set your dataset path in the sidebar or Settings page
    2. Configure processing parameters in the Settings page
    3. Navigate to the specific tool you need using the sidebar
    4. Process your data and save the results
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Developed for orthoptera bioacoustics research")

def render_settings_page():
    st.title("⚙️ Settings")
    st.write("Configure application-wide settings for all modules")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3 = st.tabs(["Paths", "Audio Processing", "Segmentation"])
    
    # Paths tab
    with tab1:
        st.header("Paths")
        
        # Dataset path with instructions
        st.session_state.dataset_path = st.text_input(
            "Dataset Path", 
            value=st.session_state.dataset_path,
            help="Path to the root folder of your orthoptera dataset"
        )
        
        # Show path helper and examples
        with st.expander("How to find your dataset path"):
            st.write("""
            ### How to copy a file path:
            
            **Windows:**
            1. Open File Explorer and navigate to your dataset folder
            2. Click in the address bar (or press Alt+D)
            3. Copy the path (Ctrl+C)
            4. Paste here (Ctrl+V)
            
            **Mac:**
            1. In Finder, navigate to your dataset folder
            2. Right-click on the folder and hold the Option key
            3. Select "Copy [folder] as Pathname"
            4. Paste here (Command+V)
            
            **Expected dataset structure:**
            ```
            dataset_root/
            ├── species_1/
            │   ├── recording1.wav
            │   ├── recording2.wav
            │   └── ...
            ├── species_2/
            │   ├── recording1.wav
            │   └── ...
            └── ...
            ```
            """)
        
        if st.session_state.dataset_path and os.path.exists(st.session_state.dataset_path):
            st.success(f"Dataset path exists: {st.session_state.dataset_path}")
            # Show dataset structure
            if os.path.isdir(st.session_state.dataset_path):
                species_folders = [f for f in os.listdir(st.session_state.dataset_path) 
                                  if os.path.isdir(os.path.join(st.session_state.dataset_path, f))]
                st.write(f"Found {len(species_folders)} species folders")
                
                # Show the species folders
                if species_folders:
                    st.write("Species folders:")
                    for folder in species_folders:
                        species_path = os.path.join(st.session_state.dataset_path, folder)
                        audio_files = [f for f in os.listdir(species_path) 
                                     if f.endswith(('.wav', '.mp3', '.flac'))]
                        st.write(f"- {folder}: {len(audio_files)} audio files")
        elif st.session_state.dataset_path:
            st.error(f"Dataset path does not exist: {st.session_state.dataset_path}")
        
        # Output path
        st.session_state.output_path = st.text_input(
            "Output Path", 
            value=st.session_state.output_path,
            help="Path where processed files will be saved"
        )
        
        if st.session_state.output_path and not os.path.exists(st.session_state.output_path):
            st.warning(f"Output path does not exist. It will be created when needed.")
        
        if not st.session_state.output_path:
            st.info("If no output path is provided, processed files will be saved in a folder next to the dataset")

    # Audio Processing tab
    with tab2:
        st.header("Audio Processing")
        
        # Normalization settings
        st.subheader("Audio Normalization")
        st.session_state.normalization = st.selectbox(
            "Normalization Method",
            ["minmax", "zscore", "peak", "none"],
            index=["minmax", "zscore", "peak", "none"].index(st.session_state.normalization),
            format_func=lambda x: {
                "minmax": "Min-Max ([-1, 1])",
                "zscore": "Z-Score",
                "peak": "Peak (0.9)",
                "none": "None (Original)"
            }.get(x),
            help="Method used to normalize audio amplitude"
        )
        
        # Add description of selected normalization method
        normalization_descriptions = {
            "minmax": "Scales the audio to the range [-1, 1] based on the maximum absolute amplitude",
            "zscore": "Standardizes the audio to have zero mean and unit variance",
            "peak": "Scales the audio to have a peak amplitude of 0.9 (good for avoiding clipping)",
            "none": "Uses the original audio without normalization"
        }
        st.info(normalization_descriptions[st.session_state.normalization])
    
    # Segmentation tab
    with tab3:
        st.header("Segmentation Settings")
        
        # Detection method
        st.subheader("Detection Method")
        st.session_state.detection_method = st.selectbox(
            "Detection Method",
            ["energy", "peaks"],
            index=["energy", "peaks"].index(st.session_state.detection_method),
            format_func=lambda x: {
                "energy": "Energy Threshold",
                "peaks": "Peak Detection"
            }.get(x),
            help="Method used to detect call segments"
        )
        
        # Common settings
        st.session_state.min_duration = st.slider(
            "Minimum Call Duration (s)", 
            min_value=0.001, 
            max_value=2.0, 
            value=st.session_state.min_duration, 
            step=0.001,
            help="Minimum duration for a valid call segment"
        )
        
        # Method-specific settings
        if st.session_state.detection_method == "energy":
            st.subheader("Energy Threshold Settings")
            
            st.session_state.min_silence = st.slider(
                "Minimum Silence Between Calls (s)", 
                min_value=0.001, 
                max_value=2.0, 
                value=st.session_state.min_silence, 
                step=0.001,
                help="Minimum silence duration between consecutive calls"
            )
            
            st.session_state.energy_threshold = st.slider(
                "Energy Threshold", 
                min_value=0.01, 
                max_value=0.5, 
                value=st.session_state.energy_threshold, 
                step=0.01,
                help="Threshold for detecting active regions (higher = less sensitive)"
            )
            
            st.info("""
            Energy threshold detection works by finding regions where the audio energy 
            exceeds the threshold. It's good for clearly separated calls with distinct 
            amplitude differences from background noise.
            """)
            
        else:  # peak detection
            st.subheader("Peak Detection Settings")
            
            st.session_state.peak_distance = st.slider(
                "Min Distance Between Peaks (frames)", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.peak_distance,
                help="Minimum distance between detected peaks (in frames)"
            )
            
            st.session_state.peak_prominence = st.slider(
                "Peak Prominence", 
                min_value=0.01, 
                max_value=0.5, 
                value=st.session_state.peak_prominence, 
                step=0.01,
                help="Required prominence of peaks (higher = stricter detection)"
            )
            
            st.session_state.peak_width = st.slider(
                "Min Peak Width (frames)", 
                min_value=0, 
                max_value=50, 
                value=st.session_state.peak_width,
                help="Minimum width of peaks in frames"
            )
            
            st.info("""
            Peak detection finds regions with significant energy peaks in the audio. 
            It's good for detecting rhythmic calls or calls with varying amplitude.
            Adjust prominence for sensitivity (lower = more sensitive).
            """)
    
    # Save settings button
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        # In a real app, you might save to a config file here

# Run the app
if __name__ == "__main__":
    main() 