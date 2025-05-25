import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from pydub import AudioSegment

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds"""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # Duration in seconds
    except Exception as e:
        st.error(f"Error getting duration for {file_path}: {e}")
        return 0

def count_durations(dataset_dir):
    """Count durations of audio files in the dataset"""
    species_durations = defaultdict(lambda: defaultdict(int))

    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Get all species folders
    species_folders = [f for f in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, f))]
    
    total_folders = len(species_folders)
    processed_folders = 0
    
    for species_folder in species_folders:
        species_path = os.path.join(dataset_dir, species_folder)
        progress_text.text(f"Processing species: {species_folder}")
        
        if os.path.isdir(species_path):
            audio_files = [f for f in os.listdir(species_path) 
                         if f.lower().endswith('.wav')]
            
            for i, audio_file in enumerate(audio_files):
                if i % 10 == 0:  # Update status less frequently to improve performance
                    progress_text.text(f"Processing {species_folder}: {i+1}/{len(audio_files)}")
                
                audio_path = os.path.join(species_path, audio_file)
                duration = get_audio_duration(audio_path)
                # Round to 2 decimal places for better grouping
                duration = round(duration, 2)
                species_durations[species_folder][duration] += 1
        
        processed_folders += 1
        progress_bar.progress(processed_folders / total_folders)
    
    progress_bar.empty()
    progress_text.empty()
    
    return species_durations

def process_dataset(dataset_dir, output_dir, target_duration, exclude_longer=True):
    """Process audio files with silence padding and optionally exclude longer files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_files = 0
    excluded_files = 0
    
    for species_folder in os.listdir(dataset_dir):
        species_path = os.path.join(dataset_dir, species_folder)
        output_species_path = os.path.join(output_dir, species_folder)

        if os.path.isdir(species_path):
            if not os.path.exists(output_species_path):
                os.makedirs(output_species_path)

            for audio_file in os.listdir(species_path):
                audio_path = os.path.join(species_path, audio_file)
                output_audio_path = os.path.join(output_species_path, audio_file)

                if audio_file.endswith('.wav'):
                    audio = AudioSegment.from_file(audio_path)
                    current_duration = len(audio) / 1000  # Get current duration in seconds
                    
                    if exclude_longer and current_duration > target_duration:
                        excluded_files += 1
                        continue  # Exclude files longer than target duration
                    
                    # Pad shorter files
                    target_length = target_duration * 1000  # Convert to ms
                    if current_duration < target_duration:
                        pad_ms = int((target_length - len(audio)) / 2)
                        silence = AudioSegment.silent(duration=pad_ms)
                        audio = silence + audio + silence
                        
                        # If odd length, add an extra ms of silence at the end to reach exact target duration
                        if len(audio) < target_length:
                            audio = audio + AudioSegment.silent(duration=target_length - len(audio))
                    
                    # Save the processed audio to the output folder
                    audio.export(output_audio_path, format="wav")
                    processed_files += 1
    
    return processed_files, excluded_files

def plot_duration_distribution(durations):
    """Plot distribution of audio durations by species"""
    # Create a dataframe from the durations dictionary
    data = []
    for species, duration_counts in durations.items():
        for duration, count in duration_counts.items():
            data.append({
                "Species": species,
                "Duration": duration,
                "Count": count
            })
    
    df = pd.DataFrame(data)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique species
    species_list = df["Species"].unique()
    
    # Create histogram-like bar chart for each species
    bar_width = 0.15
    positions = range(len(species_list))
    
    # Group durations into ranges for better visualization
    df["Duration Range"] = pd.cut(
        df["Duration"], 
        bins=[0, 0.5, 1, 2, 3, 5, 10, float('inf')],
        labels=["<0.5s", "0.5-1s", "1-2s", "2-3s", "3-5s", "5-10s", ">10s"]
    )
    
    # Aggregate by species and duration range
    pivot_df = df.pivot_table(
        index="Species", 
        columns="Duration Range", 
        values="Count", 
        aggfunc="sum",
        fill_value=0
    )
    
    # Plot as stacked bar chart
    pivot_df.plot(kind="bar", stacked=True, ax=ax)
    
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.set_title("Audio Duration Distribution by Species")
    ax.legend(title="Duration")
    
    plt.tight_layout()
    return fig

def duration_normalizer_app():
    st.header("Duration Normalizer")
    st.write("Standardize audio file durations by adding silence padding")
    
    # Get dataset path from session state
    dataset_path = st.session_state.get('dataset_path', '')
    output_path = st.session_state.get('output_path', '')
    
    if not dataset_path:
        st.warning("Please set a dataset path in the sidebar.")
        return
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    if not output_path:
        st.warning("Please set an output path in the sidebar.")
        return
    
    # Normalize settings
    st.subheader("Normalization Settings")
    
    target_duration = st.slider("Target duration (seconds):", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
    exclude_longer = st.checkbox("Exclude files longer than target duration", value=True)
    
    if st.button("Count Durations"):
        if os.path.exists(dataset_path):
            with st.spinner("Counting durations in dataset..."):
                durations = count_durations(dataset_path)
            
            st.write("### Duration Count per Species:")
            for species, counts in durations.items():
                st.write(f"#### {species}")
                
                # Create a formatted table
                duration_data = [(f"{duration:.2f}", count) for duration, count in sorted(counts.items())]
                duration_df = pd.DataFrame(duration_data, columns=["Duration (s)", "Count"])
                st.dataframe(duration_df)
                
                # Show how many files would be excluded
                if exclude_longer:
                    excluded = sum(count for duration, count in counts.items() if duration > target_duration)
                    total = sum(counts.values())
                    if total > 0:
                        st.write(f"Files that would be excluded: {excluded} ({excluded/total*100:.1f}% of {total})")
        else:
            st.error("Invalid dataset folder path.")
    
    if st.button("Process Audio Files"):
        if os.path.exists(dataset_path):
            if not os.path.exists(output_path):
                try:
                    os.makedirs(output_path)
                except Exception as e:
                    st.error(f"Could not create output directory: {e}")
                    return
            
            with st.spinner("Processing audio files..."):
                processed_files, excluded_files = process_dataset(
                    dataset_path, output_path, target_duration, exclude_longer
                )
            
            st.success(f"Processing completed! {processed_files} files processed, {excluded_files} files excluded.")
            st.write(f"Processed files are saved in: {output_path}")
        else:
            st.error("Invalid dataset folder path.") 