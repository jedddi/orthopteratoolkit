import os
import streamlit as st
import numpy as np
import pandas as pd
import time

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
from modules.call_segmentation import segment_and_visualize_file

def batch_processor_app():
    st.header("Batch Processing")
    st.write("Process multiple files and save segmented calls")
    
    # Get settings from session state
    dataset_path = st.session_state.get('dataset_path', '')
    output_path = st.session_state.get('output_path', '')
    normalization = st.session_state.get('normalization', 'minmax')
    detection_method = st.session_state.get('detection_method', 'energy')
    min_duration = st.session_state.get('min_duration', 0.05)
    min_silence = st.session_state.get('min_silence', 0.05)
    energy_threshold = st.session_state.get('energy_threshold', 0.15)
    peak_distance = st.session_state.get('peak_distance', 20)
    peak_prominence = st.session_state.get('peak_prominence', 0.1)
    peak_width = st.session_state.get('peak_width', 10)
    
    if not dataset_path:
        st.warning("Please set a dataset path in the sidebar.")
        return
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    if not output_path:
        st.warning("Please specify an output path in the sidebar to save segmented calls.")
        return
    
    # Define visualization control variables
    show_plots = st.sidebar.checkbox("Show Plots", value=True)
    max_plots = st.sidebar.number_input("Max Plots", min_value=1, value=5)
    
    # Get dataset information
    dataset_info = get_dataset_structure(dataset_path)
    
    if not dataset_info["species"]:
        st.warning("No species folders found in the dataset.")
        return
    
    # Processing options
    st.subheader("Processing Options")
    process_options = st.radio(
        "What would you like to process?",
        ["Entire Dataset", "Selected Species", "Custom Batch Processing"]
    )
    
    # Initialize session state for batch processing
    if 'batch_visualization_data' not in st.session_state:
        st.session_state.batch_visualization_data = []
    if 'batch_segments_to_process' not in st.session_state:
        st.session_state.batch_segments_to_process = {}
    
    if process_options == "Selected Species":
        selected_species = st.multiselect(
            "Select species to process",
            [species["name"] for species in dataset_info["species"]]
        )
        
        if not selected_species:
            st.warning("Please select at least one species to process.")
            return
        batch_mode = "species"
    elif process_options == "Custom Batch Processing":
        st.write("#### Custom Batch Processing")
        
        # First select species
        selected_species = st.selectbox(
            "Select species",
            [species["name"] for species in dataset_info["species"]]
        )
        
        # Then select files from that species
        species_data = next((s for s in dataset_info["species"] if s["name"] == selected_species), None)
        if species_data:
            # Option to select how many files to process at once
            batch_size = st.slider("Files per batch", 1, 10, 4)
            
            # Get all files for this species
            all_files = species_data["files"]
            
            # Calculate total number of batches
            total_batches = (len(all_files) + batch_size - 1) // batch_size
            
            # Select which batch to process
            batch_number = st.slider("Select batch number", 1, max(1, total_batches), 1)
            
            # Calculate start and end indices
            start_idx = (batch_number - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(all_files))
            
            # Get files for this batch
            batch_files = all_files[start_idx:end_idx]
            
            # Display files in this batch
            st.write(f"**Files in batch {batch_number}/{total_batches}:**")
            for idx, file in enumerate(batch_files):
                st.write(f"{idx+1}. {file}")
            
            # Custom settings for this batch
            st.write("#### Custom Settings for This Batch")
            use_custom_norm = st.checkbox("Custom normalization for this batch", value=False)
            batch_normalization = st.selectbox(
                "Batch Normalization Method",
                ["minmax", "zscore", "peak", "none"],
                index=["minmax", "zscore", "peak", "none"].index(normalization),
            ) if use_custom_norm else normalization
            
            use_custom_method = st.checkbox("Custom detection method for this batch", value=False)
            batch_detection_method = st.selectbox(
                "Batch Detection Method",
                ["energy", "peaks"],
                index=["energy", "peaks"].index(detection_method),
            ) if use_custom_method else detection_method
            
            # Custom settings based on detection method
            if batch_detection_method == "energy":
                use_custom_threshold = st.checkbox("Custom energy threshold", value=False)
                batch_energy_threshold = st.slider(
                    "Batch Energy Threshold", 0.01, 0.5, energy_threshold, 0.01
                ) if use_custom_threshold else energy_threshold
                
                use_custom_silence = st.checkbox("Custom minimum silence", value=False)
                batch_min_silence = st.slider(
                    "Batch Minimum Silence (s)", 0.05, 2.0, min_silence, 0.05
                ) if use_custom_silence else min_silence
                
                # Use default values for peak detection parameters
                batch_peak_distance = None
                batch_peak_prominence = 0.1
                batch_peak_width = None
            else:  # peaks
                use_custom_peak_distance = st.checkbox("Custom peak distance", value=False)
                batch_peak_distance = st.slider(
                    "Batch Min Distance Between Peaks", 1, 100, peak_distance or 20
                ) if use_custom_peak_distance else peak_distance
                
                use_custom_prominence = st.checkbox("Custom peak prominence", value=False)
                batch_peak_prominence = st.slider(
                    "Batch Peak Prominence", 0.01, 0.5, peak_prominence, 0.01
                ) if use_custom_prominence else peak_prominence
                
                use_custom_width = st.checkbox("Custom peak width", value=False)
                batch_peak_width = st.slider(
                    "Batch Min Peak Width", 0, 50, peak_width or 10
                ) if use_custom_width else peak_width
                
                # Use default values for energy threshold parameters
                batch_energy_threshold = energy_threshold
                batch_min_silence = min_silence
            
            use_custom_duration = st.checkbox("Custom minimum duration", value=False)
            batch_min_duration = st.slider(
                "Batch Minimum Call Duration (s)", 0.05, 2.0, min_duration, 0.05
            ) if use_custom_duration else min_duration
            
            # Store batch processing parameters
            batch_mode = "custom"
            batch_settings = {
                "species": selected_species,
                "files": batch_files,
                "normalization": batch_normalization,
                "detection_method": batch_detection_method,
                "energy_threshold": batch_energy_threshold,
                "min_silence": batch_min_silence,
                "min_duration": batch_min_duration,
                "peak_distance": batch_peak_distance,
                "peak_prominence": batch_peak_prominence,
                "peak_width": batch_peak_width
            }
        else:
            st.error(f"Species data not found for {selected_species}")
            batch_mode = None
            batch_settings = None
    else:  # Entire Dataset
        selected_species = [species["name"] for species in dataset_info["species"]]
        batch_mode = "species"
    
    # Display current processing parameters
    st.write("### Processing Parameters")
    if batch_mode == "custom" and 'batch_settings' in locals() and batch_settings:
        st.write("#### Custom Batch Settings:")
        st.write(f"Species: {batch_settings['species']}")
        st.write(f"Files in batch: {len(batch_settings['files'])}")
        st.write(f"Normalization: {batch_settings['normalization']}")
        st.write(f"Detection method: {batch_settings['detection_method']}")
        if batch_settings['detection_method'] == "energy":
            st.write(f"Energy threshold: {batch_settings['energy_threshold']}")
            st.write(f"Minimum silence: {batch_settings['min_silence']} seconds")
        else:
            st.write(f"Peak prominence: {batch_settings['peak_prominence']}")
            st.write(f"Peak distance: {batch_settings['peak_distance']} frames")
            st.write(f"Peak width: {batch_settings['peak_width']} frames")
        st.write(f"Minimum duration: {batch_settings['min_duration']} seconds")
    else:
        st.write("#### Global Settings:")
        st.write(f"Normalization: {normalization}")
        st.write(f"Detection method: {detection_method}")
        if detection_method == "energy":
            st.write(f"Energy threshold: {energy_threshold}")
            st.write(f"Minimum silence: {min_silence} seconds")
        else:
            st.write(f"Peak prominence: {peak_prominence}")
            st.write(f"Peak distance: {peak_distance} frames")
            st.write(f"Peak width: {peak_width} frames")
        st.write(f"Minimum duration: {min_duration} seconds")
    
    # --- Modified Visualization Step for Batch Processing with Segment Selection ---
    if batch_mode == "custom" and 'batch_settings' in locals() and batch_settings:
        if st.button("Visualize Sample Batch"):
            st.session_state.batch_visualization_data = []
            st.session_state.batch_segments_to_process = {}
            
            # Process all files from the current batch
            sample_files = batch_settings["files"]
            for file_index, sample_file in enumerate(sample_files):
                file_path = os.path.join(dataset_path, selected_species, sample_file)
                result = segment_and_visualize_file(
                    file_path, 
                    batch_settings["min_duration"], 
                    batch_settings["min_silence"], 
                    batch_settings["energy_threshold"],
                    normalization=batch_settings["normalization"],
                    detection_method=batch_settings["detection_method"],
                    peak_distance=batch_settings["peak_distance"],
                    peak_prominence=batch_settings["peak_prominence"],
                    peak_width=batch_settings["peak_width"]
                )
                if result:
                    st.session_state.batch_visualization_data.append({
                        "file_index": file_index,
                        "file_name": sample_file,
                        "result": result
                    })
                    # Initialize all segments as selected
                    st.session_state.batch_segments_to_process[file_index] = list(range(len(result["segments"])))
        
        # Display visualizations and segment selection options
        if st.session_state.batch_visualization_data:
            st.subheader("Sample Visualizations and Segment Selection")
            
            for viz_data in st.session_state.batch_visualization_data:
                file_index = viz_data["file_index"]
                file_name = viz_data["file_name"]
                result = viz_data["result"]
                
                st.write(f"### {selected_species} - {file_name}")
                
                # Display waveform and spectrogram
                st.plotly_chart(result["waveform_fig"], use_container_width=True)
                st.plotly_chart(result["spectrogram_fig"], use_container_width=True)
                
                # Segment selection for this file
                if len(result["segments"]) > 0:
                    st.write("#### Select segments to process:")
                    
                    # Create a list of segment options
                    segment_options = [f"Segment {i+1}" for i in range(len(result["segments"]))]
                    
                    # Display segment selection checkboxes with all pre-selected
                    selected_segments = []
                    cols = st.columns(min(4, len(result["segments"])))
                    
                    for i, segment in enumerate(result["segments"]):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            # Check if the segment was previously selected
                            is_selected = i in st.session_state.batch_segments_to_process.get(file_index, [])
                            if st.checkbox(f"Segment {i+1}", value=is_selected, key=f"seg_{file_index}_{i}"):
                                selected_segments.append(i)
                    
                    # Update the session state with the selected segments for this file
                    st.session_state.batch_segments_to_process[file_index] = selected_segments
                else:
                    st.write("No segments detected in this file.")
            
            # Add a button to select all segments for all files
            if st.button("Select All Segments"):
                for viz_data in st.session_state.batch_visualization_data:
                    file_index = viz_data["file_index"]
                    result = viz_data["result"]
                    st.session_state.batch_segments_to_process[file_index] = list(range(len(result["segments"])))
                st.rerun()
            
            # Add a button to deselect all segments for all files
            if st.button("Deselect All Segments"):
                for file_index in st.session_state.batch_segments_to_process:
                    st.session_state.batch_segments_to_process[file_index] = []
                st.rerun()
    # --- End Modified Visualization Step ---
    
    # Option to save settings for future use
    save_settings = st.checkbox("Save these settings for future reference", value=False)
    settings_name = st.text_input("Settings name", value="") if save_settings else ""
    
    if st.button("Start Batch Processing"):
        if batch_mode == "species" and not selected_species:
            st.warning("Please select at least one species to process.")
            return
        
        # Initialize counters and files to process
        if batch_mode == "custom" and 'batch_settings' in locals() and batch_settings:
            if st.session_state.batch_visualization_data:
                # Use the selected segments
                files_to_process = []
                for viz_data in st.session_state.batch_visualization_data:
                    file_index = viz_data["file_index"]
                    file_name = viz_data["file_name"]
                    
                    # Get selected segments for this file
                    selected_segments = st.session_state.batch_segments_to_process.get(file_index, [])
                    if selected_segments:  # Only process files with selected segments
                        files_to_process.append({
                            "species": selected_species,
                            "file_name": file_name,
                            "file_path": os.path.join(dataset_path, selected_species, file_name),
                            "selected_segments": selected_segments
                        })
                
                total_files = len(files_to_process)
                if total_files == 0:
                    st.warning("No files selected for processing. Please select at least one segment.")
                    return
            else:
                # Traditional batch processing (all segments)
                total_files = len(batch_settings["files"])
                files_to_process = [
                    {
                        "species": selected_species,
                        "file_name": f,
                        "file_path": os.path.join(dataset_path, selected_species, f),
                        "selected_segments": None  # None means process all segments
                    } for f in batch_settings["files"]
                ]
        else:  # species or entire dataset
            total_files = sum(s["count"] for s in dataset_info["species"] 
                             if s["name"] in selected_species)
            files_to_process = []
            for species_name in selected_species:
                species_data = next((s for s in dataset_info["species"] if s["name"] == species_name), None)
                if species_data:
                    for audio_file in species_data["files"]:
                        files_to_process.append({
                            "species": species_name,
                            "file_name": audio_file,
                            "file_path": os.path.join(dataset_path, species_name, audio_file),
                            "selected_segments": None  # None means process all segments
                        })
        
        processed_files = 0
        total_segments = 0
        visualization_data = []
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # If saving settings, store them in a dictionary
        if save_settings and settings_name:
            if batch_mode == "custom" and 'batch_settings' in locals() and batch_settings:
                settings_to_save = {
                    "name": settings_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": batch_mode,
                    "parameters": {
                        "normalization": batch_settings["normalization"],
                        "detection_method": batch_settings["detection_method"],
                        "energy_threshold": batch_settings["energy_threshold"],
                        "min_silence": batch_settings["min_silence"],
                        "min_duration": batch_settings["min_duration"],
                        "peak_distance": batch_settings["peak_distance"],
                        "peak_prominence": batch_settings["peak_prominence"],
                        "peak_width": batch_settings["peak_width"]
                    }
                }
            else:
                settings_to_save = {
                    "name": settings_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": batch_mode,
                    "parameters": {
                        "normalization": normalization,
                        "detection_method": detection_method,
                        "energy_threshold": energy_threshold,
                        "min_silence": min_silence,
                        "min_duration": min_duration,
                        "peak_distance": peak_distance,
                        "peak_prominence": peak_prominence,
                        "peak_width": peak_width
                    }
                }
            
            # Save settings to a file
            settings_dir = os.path.join(output_path, "settings")
            os.makedirs(settings_dir, exist_ok=True)
            settings_file = os.path.join(settings_dir, f"{settings_name.replace(' ', '_')}.json")
            
            try:
                import json
                with open(settings_file, 'w') as f:
                    json.dump(settings_to_save, f, indent=4)
                st.success(f"Settings saved to {settings_file}")
            except Exception as e:
                st.error(f"Error saving settings: {e}")
        
        # Process each file
        for file_info in files_to_process:
            species_name = file_info["species"]
            audio_file = file_info["file_name"]
            file_path = file_info["file_path"]
            selected_segments = file_info["selected_segments"]  # Will be None if all segments should be processed
            
            status_text.text(f"Processing {species_name} - {audio_file}")
            
            try:
                # Use batch-specific settings or global settings
                if batch_mode == "custom" and 'batch_settings' in locals() and batch_settings:
                    result = segment_and_visualize_file(
                        file_path, 
                        batch_settings["min_duration"], 
                        batch_settings["min_silence"], 
                        batch_settings["energy_threshold"],
                        normalization=batch_settings["normalization"],
                        detection_method=batch_settings["detection_method"],
                        peak_distance=batch_settings["peak_distance"],
                        peak_prominence=batch_settings["peak_prominence"],
                        peak_width=batch_settings["peak_width"]
                    )
                else:
                    result = segment_and_visualize_file(
                        file_path, min_duration, min_silence, energy_threshold,
                        normalization=normalization,
                        detection_method=detection_method,
                        peak_distance=peak_distance,
                        peak_prominence=peak_prominence,
                        peak_width=peak_width
                    )
                
                if result:
                    # Create output directory for species
                    species_output_dir = os.path.join(output_path, species_name)
                    os.makedirs(species_output_dir, exist_ok=True)
                    
                    # Filter segments based on selection if applicable
                    segments_to_save = []
                    if selected_segments is not None:
                        # Only save the selected segments
                        for i in selected_segments:
                            if i < len(result["segments"]):
                                segments_to_save.append((i, result["segments"][i]))
                    else:
                        # Save all segments
                        segments_to_save = [(i, segment) for i, segment in enumerate(result["segments"])]
                    
                    # Save filtered segments
                    for i, segment in segments_to_save:
                        segment_file = f"{os.path.splitext(audio_file)[0]}_segment_{i+1}.wav"
                        save_segment(segment["audio"], segment["sr"], species_output_dir, segment_file)
                        total_segments += 1
                    
                    # Store visualization data for sample files if enabled
                    if show_plots and len(visualization_data) < max_plots:
                        visualization_data.append({
                            "species": species_name,
                            "file": audio_file,
                            "segments_saved": len(segments_to_save),
                            "data": result
                        })
            
            except Exception as e:
                st.error(f"Error processing {file_path}: {e}")
            
            # Update progress
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
        
        # Display completion message
        status_text.text(f"Batch processing complete. Extracted {total_segments} segments from {processed_files} files.")
        
        # Display processing summary
        if visualization_data:
            st.subheader("Processing Summary")
            for viz in visualization_data:
                st.write(f"**{viz['species']} - {viz['file']}:** {viz['segments_saved']} segments saved")
        
        # Clear visualization data and selections after successful processing
        st.session_state.batch_visualization_data = []
        st.session_state.batch_segments_to_process = {} 