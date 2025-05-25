import os
import streamlit as st
import numpy as np
import librosa
import plotly.graph_objects as go

from utils.audio_processing import normalize_audio, get_audio_player, get_dataset_structure
from utils.visualization import generate_spectrogram, create_waveform_plot, create_audio_comparison_plot

def dataset_browser_app():
    st.header("Dataset Browser")
    
    # Get dataset path from session state
    dataset_path = st.session_state.get('dataset_path', '')
    if not dataset_path:
        st.warning("Please set a dataset path in the sidebar.")
        return
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Get dataset information
    dataset_info = get_dataset_structure(dataset_path)
    
    # Display dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Species", len(dataset_info["species"]))
    col2.metric("Total Files", dataset_info["total_files"])
    
    # Species selection
    st.subheader("Browse by Species")
    
    if not dataset_info["species"]:
        st.warning("No species folders found in the dataset. The dataset should have a structure with species folders containing audio files.")
        st.code("""
dataset_root/
├── species_1/
│   ├── recording1.wav
│   ├── recording2.wav
│   └── ...
├── species_2/
│   ├── recording1.wav
│   └── ...
└── ...
        """)
        return
    
    selected_species = st.selectbox(
        "Select a species to browse",
        [species["name"] for species in dataset_info["species"]]
    )
    
    # Get selected species info
    species_data = next((s for s in dataset_info["species"] if s["name"] == selected_species), None)
    
    if species_data:
        st.write(f"**{species_data['name']}**: {species_data['count']} audio files")
        
        if not species_data["files"]:
            st.warning(f"No audio files found for {selected_species}.")
            return
        
        # File selection
        selected_file = st.selectbox(
            "Select a file to preview",
            species_data["files"]
        )
        
        # Load and display selected file
        file_path = os.path.join(dataset_path, selected_species, selected_file)
        
        if os.path.exists(file_path):
            try:
                y, sr = librosa.load(file_path, sr=None)
                
                # Get normalization method from session state
                normalization = st.session_state.get('normalization', 'minmax')
                
                # Apply normalization if selected
                if normalization != 'none':
                    y_normalized = normalize_audio(y, method=normalization)
                    
                    # Show comparison between original and normalized
                    st.write("### Normalization Comparison")
                    
                    # Create comparison plot
                    fig = create_audio_comparison_plot(y, y_normalized, sr, normalization)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display file info with normalization
                    duration = librosa.get_duration(y=y, sr=sr)
                    st.write(f"Duration: {duration:.2f} seconds | Sample rate: {sr} Hz")
                    st.write(f"Original amplitude range: [{y.min():.2f}, {y.max():.2f}]")
                    st.write(f"Normalized amplitude range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
                    
                    # Audio players for comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Audio")
                        st.markdown(get_audio_player(y, sr), unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"Normalized Audio ({normalization})")
                        st.markdown(get_audio_player(y_normalized, sr), unsafe_allow_html=True)
                    
                    # Use normalized audio for spectrogram
                    y_display = y_normalized
                else:
                    # Display file info without normalization
                    duration = librosa.get_duration(y=y, sr=sr)
                    st.write(f"Duration: {duration:.2f} seconds | Sample rate: {sr} Hz")
                    st.write(f"Amplitude range: [{y.min():.2f}, {y.max():.2f}]")
                    
                    # Audio player
                    st.write("### Audio Preview")
                    st.markdown(get_audio_player(y, sr), unsafe_allow_html=True)
                    
                    # Use original audio for display
                    y_display = y
                
                # Waveform
                st.write("### Waveform")
                waveform_fig = create_waveform_plot(y_display, sr)
                st.plotly_chart(waveform_fig, use_container_width=True)
                
                # Spectrogram
                st.write("### Spectrogram")
                spec_fig = generate_spectrogram(y_display, sr, f"Spectrogram - {selected_file}")
                st.plotly_chart(spec_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    else:
        st.error(f"Species data not found for {selected_species}") 