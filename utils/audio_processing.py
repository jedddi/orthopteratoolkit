import os
import numpy as np
import librosa
import soundfile as sf
import io
import base64
from scipy import stats

def normalize_audio(y, method='minmax'):
    """
    Normalize audio amplitude using different methods.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    method : str
        Normalization method: 'minmax', 'zscore', or 'peak'
        
    Returns:
    --------
    normalized_y : ndarray
        Normalized audio signal
    """
    if method == 'minmax':
        # Min-max normalization to range [-1, 1]
        if np.max(np.abs(y)) > 0:
            return y / np.max(np.abs(y))
        return y
    elif method == 'zscore':
        # Z-score normalization
        if np.std(y) > 0:
            return stats.zscore(y)
        return y
    elif method == 'peak':
        # Peak normalization (similar to minmax but uses a target peak value)
        target_peak = 0.9  # Target peak amplitude (slightly below clipping)
        if np.max(np.abs(y)) > 0:
            return y * (target_peak / np.max(np.abs(y)))
        return y
    else:
        return y  # Return original if method not recognized

def get_audio_player(audio_data, sr):
    """Create an HTML audio player for the audio data"""
    # Save audio to a BytesIO object
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
    
    # Create HTML audio player
    audio_html = f'<audio controls><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    return audio_html

def save_segment(y, sr, output_path, file_name):
    """Save audio segment to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(os.path.join(output_path, file_name), y, sr)
    return os.path.join(output_path, file_name)

def get_dataset_structure(dataset_path):
    """Get the structure of the dataset: species and files"""
    dataset_info = {"species": [], "files": [], "total_files": 0}
    
    try:
        species_folders = [f for f in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, f))]
        
        for species in species_folders:
            species_path = os.path.join(dataset_path, species)
            audio_files = [f for f in os.listdir(species_path) 
                         if f.endswith(('.wav', '.mp3', '.flac'))]
            
            dataset_info["species"].append({
                "name": species,
                "count": len(audio_files),
                "files": audio_files
            })
            dataset_info["files"].extend([os.path.join(species, file) for file in audio_files])
            dataset_info["total_files"] += len(audio_files)
    except Exception as e:
        print(f"Error reading dataset structure: {e}")
    
    return dataset_info

def detect_call_segments_energy(energy, frame_time, threshold, min_duration, min_silence):
    """
    Detect call segments based on energy envelope.
    
    Parameters:
    -----------
    energy : ndarray
        Energy envelope of the audio
    frame_time : ndarray
        Time points corresponding to energy frames
    threshold : float
        Energy threshold for detection
    min_duration : float
        Minimum duration (in seconds) for a valid call segment
    min_silence : float
        Minimum silence duration (in seconds) between calls
        
    Returns:
    --------
    segments : list
        List of tuples (start_time, end_time) for each detected segment
    """
    # Find regions above threshold
    active_frames = energy > threshold
    
    # Convert to time segments
    active_regions = []
    in_active = False
    start_idx = 0
    
    for i, active in enumerate(active_frames):
        if active and not in_active:
            # Start of active region
            in_active = True
            start_idx = i
        elif not active and in_active:
            # End of active region
            in_active = False
            active_regions.append((frame_time[start_idx], frame_time[i]))
    
    # Handle case where file ends during active region
    if in_active:
        active_regions.append((frame_time[start_idx], frame_time[-1]))
    
    # Merge segments that are close together (handling recurring calls)
    merged_regions = []
    if active_regions:
        current_start, current_end = active_regions[0]
        
        for start, end in active_regions[1:]:
            if start - current_end < min_silence:
                # Merge with previous segment
                current_end = end
            else:
                # Add previous segment and start a new one
                merged_regions.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last segment
        merged_regions.append((current_start, current_end))
    
    # Filter segments that are too short
    segments = [(start, end) for start, end in merged_regions 
               if end - start >= min_duration]
    
    return segments

def detect_call_segments_peaks(y, sr, frame_time, energy, distance=None, prominence=0.1, width=None, min_duration=0.2):
    """
    Detect call segments using peak detection algorithms.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sample rate
    frame_time : ndarray
        Time points corresponding to energy frames
    energy : ndarray
        Energy envelope of the audio
    distance : int or None
        Minimum distance between peaks (in frames)
    prominence : float
        Required prominence of peaks
    width : int or None
        Required width of peaks
    min_duration : float
        Minimum duration (in seconds) for a valid call segment
        
    Returns:
    --------
    segments : list
        List of tuples (start_time, end_time) for each detected segment
    peak_info : dict
        Dictionary containing peak indices and properties
    """
    from scipy.signal import find_peaks
    
    # Set default distance if not provided
    if distance is None:
        distance = int(sr * 0.1 / 512)  # 0.1s in frames (assuming hop_length=512)
        
    # Find peaks in the energy envelope
    peaks, properties = find_peaks(
        energy,
        distance=distance,
        prominence=prominence,
        width=width
    )
    
    # Convert peak indices to time
    peak_times = frame_time[peaks]
    
    # Extract segments around peaks
    segments = []
    
    # Approximate the width of peaks in time
    if 'left_bases' in properties and 'right_bases' in properties:
        for i, peak_idx in enumerate(peaks):
            # Get left and right edges of the peak
            left_idx = properties['left_bases'][i]
            right_idx = properties['right_bases'][i]
            
            # Convert to time
            if left_idx < len(frame_time) and right_idx < len(frame_time):
                start_time = frame_time[left_idx]
                end_time = frame_time[right_idx]
                
                # Check if segment is long enough
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
    
    # If no segments were created using bases, use peak prominences as an approximation
    if not segments and 'prominences' in properties:
        for i, peak_time in enumerate(peak_times):
            # Approximate a segment duration based on prominence
            # Higher prominence typically means longer/stronger calls
            approx_duration = min(0.5, 0.1 + properties['prominences'][i])
            
            start_time = peak_time - approx_duration/2
            end_time = peak_time + approx_duration/2
            
            # Ensure start time is not negative
            start_time = max(0, start_time)
            
            # Check if segment is long enough
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
    
    # Return both segments and peak information
    peak_info = {
        'indices': peaks,
        'times': peak_times,
        'properties': properties
    }
    
    return segments, peak_info 