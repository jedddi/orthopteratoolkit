import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import shutil
import time
from pathlib import Path

def is_silent(audio_path, threshold=0.001, min_duration=0.1):
    """
    Check if an audio file is silent based on amplitude threshold.
    
    Parameters:
    audio_path (str): Path to the audio file
    threshold (float): RMS amplitude threshold below which audio is considered silent
    min_duration (float): Minimum duration of non-silent audio required (in seconds)
    
    Returns:
    bool: True if audio is silent, False otherwise
    dict: Additional stats about the audio file
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Get file duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Get max RMS value
        max_rms = np.max(rms)
        mean_rms = np.mean(rms)
        
        # Calculate percentage of frames above threshold
        frames_above_threshold = np.sum(rms > threshold)
        total_frames = len(rms)
        percent_active = (frames_above_threshold / total_frames) * 100
        
        # Non-silent duration estimation
        frame_duration = duration / total_frames
        non_silent_duration = frames_above_threshold * frame_duration
        
        # Check if the file is silent
        is_silent = max_rms < threshold or non_silent_duration < min_duration
        
        stats = {
            "duration": duration,
            "max_rms": max_rms,
            "mean_rms": mean_rms,
            "percent_active": percent_active,
            "non_silent_duration": non_silent_duration
        }
        
        return is_silent, stats
    
    except Exception as e:
        st.error(f"Error processing {audio_path}: {str(e)}")
        return True, {"error": str(e)}

def scan_directory(directory, extensions=['.wav', '.mp3', '.flac', '.ogg'], threshold=0.001, min_duration=0.1):
    """
    Scan a directory for silent audio files.
    
    Parameters:
    directory (str): Directory path to scan
    extensions (list): List of audio file extensions to check
    threshold (float): RMS amplitude threshold
    min_duration (float): Minimum non-silent duration required
    
    Returns:
    pd.DataFrame: DataFrame containing scan results
    """
    results = []
    
    # Get all audio files recursively
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each file
    for i, file_path in enumerate(audio_files):
        status_text.text(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        
        # Get species folder name
        rel_path = os.path.relpath(file_path, directory)
        species = rel_path.split(os.sep)[0] if os.sep in rel_path else "Unknown"
        
        silent, stats = is_silent(file_path, threshold, min_duration)
        
        results.append({
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "species": species,
            "is_silent": silent,
            "duration": stats.get("duration", 0),
            "max_rms": stats.get("max_rms", 0),
            "mean_rms": stats.get("mean_rms", 0),
            "percent_active": stats.get("percent_active", 0),
            "non_silent_duration": stats.get("non_silent_duration", 0)
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(audio_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def display_audio_stats(results_df):
    """Display summary statistics about the scanned audio files"""
    st.subheader("Dataset Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", len(results_df))
        
    with col2:
        silent_count = results_df['is_silent'].sum()
        st.metric("Silent Files", f"{silent_count} ({silent_count/len(results_df)*100:.1f}%)")
        
    with col3:
        species_count = results_df['species'].nunique()
        st.metric("Species Count", species_count)
    
    # Species breakdown
    st.subheader("Files by Species")
    species_counts = results_df.groupby('species').agg({
        'is_silent': ['count', 'sum'],
        'duration': 'mean'
    })
    species_counts.columns = ['Total Files', 'Silent Files', 'Avg Duration (s)']
    species_counts['Silent %'] = (species_counts['Silent Files'] / species_counts['Total Files'] * 100).round(1)
    st.dataframe(species_counts)
    
    # Distribution of RMS values
    st.subheader("RMS Energy Distribution")
    fig = plot_rms_distribution(results_df)
    st.pyplot(fig)

def remove_file(file_path, trash_dir=None):
    """
    Remove a file either by moving it to trash or deleting it permanently
    
    Parameters:
    file_path (str): Path to the file to remove
    trash_dir (str, optional): Directory to move file to instead of deleting
    
    Returns:
    bool: True if successful, False otherwise
    str: Message about the operation result
    """
    try:
        if trash_dir:
            # Create trash directory if it doesn't exist
            os.makedirs(trash_dir, exist_ok=True)
            
            # Create species subdirectory in trash if needed
            rel_path = os.path.dirname(os.path.relpath(file_path, os.path.dirname(trash_dir)))
            species_trash_dir = os.path.join(trash_dir, rel_path)
            os.makedirs(species_trash_dir, exist_ok=True)
            
            # Move file to trash
            dest_path = os.path.join(species_trash_dir, os.path.basename(file_path))
            shutil.move(file_path, dest_path)
            return True, f"Moved to {dest_path}"
        else:
            # Delete file permanently
            os.remove(file_path)
            return True, "Deleted permanently"
    except Exception as e:
        return False, f"Error: {str(e)}"

def silence_detector_app():
    st.title("🔇 Orthoptera Silent Audio Detector")
    st.write("Scan your orthoptera dataset to find silent audio files across different species.")
    
    # Get dataset path from session state
    dataset_path = st.session_state.get('dataset_path', '')
    
    if not dataset_path:
        st.warning("Please set a dataset path in the sidebar.")
        return
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Initialize session state for managing the UI flow
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "scan"  # Options: scan, review
    if 'files_to_remove' not in st.session_state:
        st.session_state.files_to_remove = set()
    if 'removed_files' not in st.session_state:
        st.session_state.removed_files = set()
    
    # Sidebar settings
    st.sidebar.header("Detection Settings")
    
    threshold = st.sidebar.slider(
        "Silence Threshold (RMS)", 
        min_value=0.0001, 
        max_value=0.01, 
        value=0.001,
        format="%.4f"
    )
    
    min_duration = st.sidebar.slider(
        "Minimum Non-Silent Duration (seconds)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.1,
        format="%.2f"
    )
    
    extensions = st.sidebar.multiselect(
        "Audio File Extensions",
        options=['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a'],
        default=['.wav', '.mp3', '.flac', '.ogg']
    )
    
    use_trash = st.sidebar.checkbox("Move to trash instead of deleting", value=True)
    if use_trash:
        trash_dir = st.sidebar.text_input("Trash Directory", "trash_silent_files")
        if not os.path.isabs(trash_dir) and dataset_path:
            trash_dir = os.path.join(os.path.dirname(dataset_path), trash_dir)
    else:
        trash_dir = None
    
    # SCAN VIEW
    if st.session_state.current_view == "scan":
        scan_button = st.button("Scan Dataset", type="primary")
        
        if scan_button:
            with st.spinner("Scanning audio files..."):
                start_time = time.time()
                results_df = scan_directory(
                    dataset_path, 
                    extensions=extensions,
                    threshold=threshold,
                    min_duration=min_duration
                )
                scan_time = time.time() - start_time
            
            st.success(f"Scan completed in {scan_time:.2f} seconds")
            
            # Store results in session state
            st.session_state.results_df = results_df
            
            # Display summary statistics
            if not results_df.empty:
                st.subheader("Dataset Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Files", len(results_df))
                    
                with col2:
                    silent_count = results_df['is_silent'].sum()
                    st.metric("Silent Files", f"{silent_count} ({silent_count/len(results_df)*100:.1f}%)")
                    
                with col3:
                    species_count = results_df['species'].nunique()
                    st.metric("Species Count", species_count)
                
                # Species breakdown
                st.subheader("Files by Species")
                species_counts = results_df.groupby('species').agg({
                    'is_silent': ['count', 'sum'],
                    'duration': 'mean'
                })
                species_counts.columns = ['Total Files', 'Silent Files', 'Avg Duration (s)']
                species_counts['Silent %'] = (species_counts['Silent Files'] / species_counts['Total Files'] * 100).round(1)
                st.dataframe(species_counts)
                
                # Display results table
                st.subheader("Silent Files")
                silent_files = results_df[results_df['is_silent']]
                if len(silent_files) > 0:
                    st.dataframe(silent_files[['file_path', 'species', 'duration', 'max_rms', 'percent_active']])
                    
                    # Export options
                    st.download_button(
                        label="Download Silent Files Report (CSV)",
                        data=silent_files.to_csv(index=False),
                        file_name="silent_files_report.csv",
                        mime="text/csv"
                    )
                    
                    # Option to review files
                    if st.button("Review Silent Files"):
                        st.session_state.current_view = "review"
                        st.rerun()
                else:
                    st.success("No silent files found in the dataset!")
    
    # REVIEW VIEW
    elif st.session_state.current_view == "review":
        if st.session_state.results_df is None:
            st.error("No scan results available. Please scan your dataset first.")
            st.session_state.current_view = "scan"
            st.rerun()
            return
        
        # Get silent files
        silent_files = st.session_state.results_df[st.session_state.results_df['is_silent']].copy()
        
        if len(silent_files) == 0:
            st.success("No silent files found in the dataset!")
            st.session_state.current_view = "scan"
            st.rerun()
            return
        
        # Mark files that have been removed
        silent_files['status'] = silent_files['file_path'].apply(
            lambda x: "Removed" if x in st.session_state.removed_files else "Active"
        )
        active_silent_files = silent_files[silent_files['status'] == "Active"]
        
        # Show removal stats
        st.subheader("Silent Files Review")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Silent Files", len(silent_files))
        with col2:
            st.metric("Files Removed", len(st.session_state.removed_files))
        
        # Group by species
        st.subheader("Select Species to Review")
        species_list = active_silent_files['species'].unique()
        if len(species_list) == 0:
            st.success("All silent files have been reviewed!")
            if st.button("Return to Scan"):
                st.session_state.current_view = "scan"
                st.rerun()
            return
        
        selected_species = st.selectbox("Select Species", species_list)
        species_files = active_silent_files[active_silent_files['species'] == selected_species]
        
        # Review files for selected species
        st.subheader(f"Silent Files for {selected_species}")
        st.write(f"Found {len(species_files)} silent files")
        
        # Create tabs for individual and batch review
        tab1, tab2 = st.tabs(["Individual Review", "Batch Review"])
        
        # Individual Review Tab
        with tab1:
            if len(species_files) > 0:
                file_indices = species_files.index.tolist()
                
                # Use a slider to navigate through files
                if len(file_indices) > 1:
                    file_index = st.slider("File Navigation", 0, len(file_indices)-1, 0)
                else:
                    file_index = 0
                
                current_file = species_files.iloc[file_index]
                file_path = current_file['file_path']
                
                # Display file info and audio player
                st.subheader(f"File {file_index+1} of {len(species_files)}: {current_file['filename']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Path:** {file_path}")
                    st.write(f"**Duration:** {current_file['duration']:.2f} seconds")
                    st.write(f"**Max RMS:** {current_file['max_rms']:.6f}")
                    st.write(f"**Active Content:** {current_file['percent_active']:.2f}%")
                
                with col2:
                    # Audio player
                    try:
                        st.audio(file_path)
                    except Exception as e:
                        st.error(f"Error playing audio: {str(e)}")
                
                # Remove button
                if st.button(f"Remove This File", key=f"remove_{file_path}"):
                    success, message = remove_file(file_path, trash_dir)
                    if success:
                        st.session_state.removed_files.add(file_path)
                        st.success(f"File removed: {message}")
                        # Refresh the view
                        st.rerun()
                    else:
                        st.error(message)
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if file_index > 0:
                        if st.button("Previous File"):
                            st.rerun()
                
                with col2:
                    if file_index < len(file_indices) - 1:
                        if st.button("Next File"):
                            st.rerun()
            else:
                st.write("No silent files found for this species.")
        
        # Batch Review Tab
        with tab2:
            if len(species_files) > 0:
                st.write("Select files to remove:")
                
                # Select all option
                select_all = st.checkbox("Select All Files", key=f"select_all_{selected_species}")
                
                # Create checkboxes for each file
                selected_files = []
                for idx, row in species_files.iterrows():
                    file_path = row['file_path']
                    filename = row['filename']
                    
                    # Create a unique key for each checkbox
                    checkbox_key = f"check_{idx}"
                    
                    # If select all is checked, pre-select all checkboxes
                    if select_all:
                        is_selected = st.checkbox(
                            f"{filename} (Duration: {row['duration']:.2f}s, RMS: {row['max_rms']:.6f})",
                            value=True,
                            key=checkbox_key
                        )
                    else:
                        is_selected = st.checkbox(
                            f"{filename} (Duration: {row['duration']:.2f}s, RMS: {row['max_rms']:.6f})",
                            key=checkbox_key
                        )
                    
                    if is_selected:
                        selected_files.append(file_path)
                
                # Batch removal button
                if selected_files:
                    if st.button(f"Remove {len(selected_files)} Selected Files", key="batch_remove"):
                        removed_count = 0
                        errors = []
                        
                        for file_path in selected_files:
                            success, message = remove_file(file_path, trash_dir)
                            if success:
                                st.session_state.removed_files.add(file_path)
                                removed_count += 1
                            else:
                                errors.append(f"{os.path.basename(file_path)}: {message}")
                        
                        if removed_count > 0:
                            st.success(f"Removed {removed_count} files")
                        
                        if errors:
                            st.error("Errors occurred:")
                            for error in errors:
                                st.write(error)
                        
                        # Refresh the view
                        st.rerun()
            else:
                st.write("No silent files found for this species.")
        
        # Return button
        if st.button("Return to Scan Results"):
            st.session_state.current_view = "scan"
            st.rerun() 