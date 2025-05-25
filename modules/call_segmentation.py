import os
import streamlit as st
import numpy as np
import librosa
import pandas as pd
import plotly.graph_objects as go

from utils.audio_processing import (
    normalize_audio, 
    get_audio_player, 
    get_dataset_structure,
    detect_call_segments_energy,
    detect_call_segments_peaks,
    save_segment
)
from utils.visualization import (
    generate_spectrogram, 
    visualize_audio_with_segments
)

def segment_and_visualize_file(file_path, min_duration, min_silence, energy_threshold, 
                              normalization='minmax', detection_method='energy',
                              peak_distance=None, peak_prominence=0.1, peak_width=None):
    """Segment a single file and return visualization data"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Store original audio for comparison and segmentation
        y_original = y.copy()

        # Apply normalization
        y_normalized = normalize_audio(y, method=normalization) if normalization != 'none' else y

        # Compute energy envelope (on normalized audio)
        energy = librosa.feature.rms(y=y_normalized, frame_length=1024, hop_length=512)[0]

        # Normalize energy to [0, 1] range
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy

        # Convert frames to time
        frame_time = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=512)

        # Find segments based on selected method
        peak_info = None
        if detection_method == 'energy':
            segments = detect_call_segments_energy(energy, frame_time, energy_threshold, 
                                                min_duration, min_silence)
            waveform_fig = visualize_audio_with_segments(
                y_normalized, sr, energy, frame_time, segments, energy_threshold
            )
        elif detection_method == 'peaks':
            segments, peak_info = detect_call_segments_peaks(
                y_normalized, sr, frame_time, energy,
                distance=peak_distance,
                prominence=peak_prominence,
                width=peak_width,
                min_duration=min_duration
            )
            # Get peak times and values for visualization
            peak_times = peak_info['times'] if peak_info else None
            peak_values = energy[peak_info['indices']] * np.max(np.abs(y_normalized)) * 0.8 if peak_info else None
            
            waveform_fig = visualize_audio_with_segments(
                y_normalized, sr, energy, frame_time, segments, 
                peaks=peak_times, peak_values=peak_values
            )
        else:
            # Default to energy-based method
            segments = detect_call_segments_energy(energy, frame_time, energy_threshold, 
                                                min_duration, min_silence)
            waveform_fig = visualize_audio_with_segments(
                y_normalized, sr, energy, frame_time, segments, energy_threshold
            )

        # Create spectrogram
        spectrogram = generate_spectrogram(y_normalized, sr, 
                                         title=f"Spectrogram - {os.path.basename(file_path)}")

        # Extract segments FROM ORIGINAL AUDIO using the detected segment times
        segment_data = []
        for i, (start_time, end_time) in enumerate(segments):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Ensure we don't exceed array bounds
            if end_sample > len(y_original):
                end_sample = len(y_original)

            segment_audio = y_original[start_sample:end_sample]  # Extract from original audio
            segment_data.append({
                "id": i + 1,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "audio": segment_audio,
                "sr": sr
            })

        return {
            "waveform_fig": waveform_fig,
            "spectrogram_fig": spectrogram,
            "segments": segment_data,
            "duration": duration,
            "sample_rate": sr,
            "full_audio": y_original,  # Use original for playback
            "original_audio": y_original,  # Ensure this key exists
            "normalized_audio": y_normalized,
            "detection_method": detection_method
        }

    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return None

def call_segmentation_app():
    st.header("Call Segmentation")
    st.write("Segment and validate individual call recordings")
    
    # Get dataset path from session state
    dataset_path = st.session_state.get('dataset_path', '')
    output_path = st.session_state.get('output_path', '')
    
    if not dataset_path:
        st.warning("Please set a dataset path in the sidebar.")
        return
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Get dataset information
    dataset_info = get_dataset_structure(dataset_path)
    
    if not dataset_info["species"]:
        st.warning("No species folders found in the dataset.")
        return
    
    # File selection
    st.subheader("Select File")
    
    # First select species
    selected_species = st.selectbox(
        "Select species",
        [species["name"] for species in dataset_info["species"]]
    )
    
    # Then select file
    species_data = next((s for s in dataset_info["species"] if s["name"] == selected_species), None)
    
    if not species_data or not species_data["files"]:
        st.warning(f"No audio files found for {selected_species}.")
        return
        
    selected_file = st.selectbox(
        "Select file",
        species_data["files"]
    )
    
    file_path = os.path.join(dataset_path, selected_species, selected_file)
    
    # Get segmentation parameters from session state
    detection_method = st.session_state.get('detection_method', 'energy')
    normalization = st.session_state.get('normalization', 'minmax')
    min_duration = st.session_state.get('min_duration', 0.05)
    min_silence = st.session_state.get('min_silence', 0.05)
    energy_threshold = st.session_state.get('energy_threshold', 0.15)
    peak_distance = st.session_state.get('peak_distance', 20)
    peak_prominence = st.session_state.get('peak_prominence', 0.1)
    peak_width = st.session_state.get('peak_width', 10)
    
    # Option to override parameters
    with st.expander("Override Segmentation Parameters"):
        use_custom_params = st.checkbox("Use custom parameters for this file", value=False)
        
        if use_custom_params:
            col1, col2 = st.columns(2)
            
            with col1:
                custom_detection = st.selectbox(
                    "Detection Method",
                    ["energy", "peaks"],
                    index=0 if detection_method == "energy" else 1,
                    format_func=lambda x: "Energy Threshold" if x == "energy" else "Peak Detection"
                )
                
                custom_normalization = st.selectbox(
                    "Normalization Method",
                    ["minmax", "zscore", "peak", "none"],
                    index=["minmax", "zscore", "peak", "none"].index(normalization),
                    format_func=lambda x: {
                        "minmax": "Min-Max ([-1, 1])",
                        "zscore": "Z-Score",
                        "peak": "Peak (0.9)",
                        "none": "None (Original)"
                    }.get(x)
                )
            
            with col2:
                custom_min_duration = st.slider(
                    "Minimum Call Duration (s)",
                    min_value=0.001, max_value=2.0, value=min_duration, step=0.001
                )
                
                if custom_detection == "energy":
                    custom_min_silence = st.slider(
                        "Minimum Silence Between Calls (s)",
                        min_value=0.001, max_value=2.0, value=min_silence, step=0.001
                    )
                    custom_energy_threshold = st.slider(
                        "Energy Threshold",
                        0.01, 0.5, energy_threshold, 0.01
                    )
                    custom_peak_distance = None
                    custom_peak_prominence = 0.1
                    custom_peak_width = None
                else:  # peak detection
                    custom_min_silence = min_silence  # default value
                    custom_energy_threshold = energy_threshold  # default value
                    custom_peak_distance = st.slider(
                        "Min Distance Between Peaks (frames)",
                        1, 100, peak_distance
                    )
                    custom_peak_prominence = st.slider(
                        "Peak Prominence",
                        0.01, 0.5, peak_prominence, 0.01
                    )
                    custom_peak_width = st.slider(
                        "Min Peak Width (frames)",
                        0, 50, peak_width
                    )
        else:
            # Use global parameters
            custom_detection = detection_method
            custom_normalization = normalization
            custom_min_duration = min_duration
            custom_min_silence = min_silence
            custom_energy_threshold = energy_threshold
            custom_peak_distance = peak_distance
            custom_peak_prominence = peak_prominence
            custom_peak_width = peak_width
    
    if st.button("Analyze File"):
        with st.spinner("Analyzing audio..."):
            result = segment_and_visualize_file(
                file_path, 
                custom_min_duration, 
                custom_min_silence, 
                custom_energy_threshold,
                normalization=custom_normalization,
                detection_method=custom_detection,
                peak_distance=custom_peak_distance,
                peak_prominence=custom_peak_prominence,
                peak_width=custom_peak_width
            )
            
            if result:
                # Display processing information
                st.write(f"### Processing Information")
                st.write(f"Normalization: {custom_normalization}")
                st.write(f"Detection method: {custom_detection}")
                
                # Display waveform with segments
                st.write("### Waveform with Segments")
                st.plotly_chart(result["waveform_fig"], use_container_width=True)
                
                # Display spectrogram
                st.write("### Spectrogram")
                st.plotly_chart(result["spectrogram_fig"], use_container_width=True)
                
                # Audio comparison if normalized
                if custom_normalization != 'none':
                    st.write("### Audio Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original Audio")
                        st.markdown(get_audio_player(result["original_audio"], result["sample_rate"]), 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"Normalized Audio ({custom_normalization})")
                        st.markdown(get_audio_player(result["normalized_audio"], result["sample_rate"]), 
                                  unsafe_allow_html=True)
                else:
                    # Full audio preview (only one version)
                    st.write("### Full Audio")
                    st.markdown(get_audio_player(result["full_audio"], result["sample_rate"]), 
                              unsafe_allow_html=True)
                
                # Display segments
                st.write(f"### Detected Segments ({len(result['segments'])})")
                
                if result["segments"]:
                    segments_df = pd.DataFrame([
                        {"Segment": f"Segment {s['id']}", 
                         "Start (s)": f"{s['start']:.2f}", 
                         "End (s)": f"{s['end']:.2f}", 
                         "Duration (s)": f"{s['duration']:.2f}"}
                        for s in result["segments"]
                    ])
                    st.dataframe(segments_df)
                    
                    # Preview individual segments
                    st.write("### Segment Preview")
                    
                    # Calculate how many columns to use based on number of segments
                    num_cols = min(4, len(result["segments"]))
                    cols = st.columns(num_cols)
                    
                    for i, segment in enumerate(result["segments"]):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            st.write(f"**Segment {segment['id']}**")
                            st.markdown(get_audio_player(segment['audio'], segment['sr']), 
                                      unsafe_allow_html=True)
                            
                            # Display validation options
                            valid = st.checkbox(f"Valid call", value=True, key=f"valid_{i}")
                            
                            if valid and output_path:
                                if st.button(f"Save segment", key=f"save_{i}"):
                                    segment_file = f"{os.path.splitext(selected_file)[0]}_segment_{i+1}.wav"
                                    species_folder = os.path.join(output_path, selected_species)
                                    file_saved = save_segment(segment['audio'], segment['sr'], 
                                                             species_folder, segment_file)
                                    st.success(f"Saved segment to {file_saved}")
                            elif valid and not output_path:
                                st.warning("Set an output path in the sidebar to save segments")
                else:
                    st.warning("No segments detected with current settings. Try adjusting the parameters.")
    
    if not output_path:
        st.warning("Set an output path in the sidebar to save segments.") 