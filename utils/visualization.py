import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import librosa

def generate_spectrogram(y, sr, title="Spectrogram"):
    """Generate a spectrogram using plotly for better interactivity"""
    # Calculate spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Create Plotly figure
    fig = px.imshow(
        D, 
        aspect='auto',
        origin='lower',
        labels=dict(x="Time", y="Frequency", color="dB"),
        title=title
    )
    
    # Improve layout
    fig.update_layout(
        coloraxis_colorbar=dict(title="dB"),
        xaxis_title="Time (frames)",
        yaxis_title="Frequency (bins)"
    )
    
    return fig

def create_waveform_plot(y, sr, color='blue', title="Waveform"):
    """Create a waveform plot using plotly"""
    time = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, line=dict(color=color)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude", 
        height=300
    )
    return fig

def create_waveform_with_segments(y, sr, segments, energy=None, frame_time=None, 
                               threshold=None, peak_info=None, normalization='none'):
    """Create a waveform plot with detected segments highlighted"""
    fig = go.Figure()
    
    # Add time axis
    time = np.arange(len(y)) / sr
    
    # Add original waveform
    fig.add_trace(go.Scatter(x=time, y=y, name='Waveform', 
                         line=dict(color='blue')))
    
    # Add energy envelope if provided
    if energy is not None and frame_time is not None:
        # Scale energy to be visible alongside waveform
        energy_scaled = energy * np.max(np.abs(y)) * 0.8
        fig.add_trace(go.Scatter(x=frame_time, y=energy_scaled, 
                             name='Energy', line=dict(color='green')))
        
        # Add threshold line if provided
        if threshold is not None:
            threshold_line = np.ones_like(frame_time) * threshold * np.max(np.abs(y)) * 0.8
            fig.add_trace(go.Scatter(x=frame_time, y=threshold_line, 
                                  name='Threshold', line=dict(color='red', dash='dash')))
    
    # Add peak markers if provided
    if peak_info is not None and energy is not None:
        peak_times = peak_info['times']
        peak_values = energy[peak_info['indices']] * np.max(np.abs(y)) * 0.8
        fig.add_trace(go.Scatter(x=peak_times, y=peak_values, 
                             mode='markers', name='Peaks', 
                             marker=dict(color='red', size=8)))
    
    # Add segments as vertical regions
    for i, (start, end) in enumerate(segments):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below", line_width=0,
            annotation_text=f"Segment {i+1}"
        )
    
    # Set layout
    fig.update_layout(
        title="Waveform with Detected Segments",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(x=0.01, y=0.99),
        height=400
    )
    
    return fig

def create_audio_comparison_plot(y_original, y_normalized, sr, normalization_method="minmax"):
    """Create a comparison plot between original and normalized audio"""
    fig = go.Figure()
    time = np.arange(len(y_original)) / sr
    
    fig.add_trace(go.Scatter(x=time, y=y_original, name='Original', 
                          line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=time, y=y_normalized, 
                          name=f'Normalized ({normalization_method})', 
                          line=dict(color='blue')))
    
    fig.update_layout(
        title="Audio Normalization Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude", 
        height=300
    )
    return fig

def visualize_audio_with_segments(y, sr, energy, frame_time, segments, 
                                energy_threshold=None, peaks=None, peak_values=None):
    """Visualize audio with detected segments and optionally energy envelope and peaks"""
    # Create audio visualization
    fig = go.Figure()

    # Add waveform
    time = np.arange(len(y)) / sr
    fig.add_trace(go.Scatter(x=time, y=y, name='Waveform', 
                           line=dict(color='blue')))

    # Add energy envelope (scaled for visibility)
    if energy is not None and frame_time is not None:
        energy_scaled = energy * np.max(np.abs(y)) * 0.8
        fig.add_trace(go.Scatter(x=frame_time, y=energy_scaled, name='Energy', 
                               line=dict(color='green')))

    # Add threshold line for energy method
    if energy_threshold is not None:
        threshold_line = np.ones_like(frame_time) * energy_threshold * np.max(np.abs(y)) * 0.8
        fig.add_trace(go.Scatter(x=frame_time, y=threshold_line, name='Threshold',
                               line=dict(color='red', dash='dash')))

    # Add peaks if provided
    if peaks is not None and peak_values is not None:
        fig.add_trace(go.Scatter(x=peaks, y=peak_values, mode='markers',
                               name='Detected Peaks', marker=dict(color='red', size=8)))

    # Add segments as vertical regions
    for i, (start, end) in enumerate(segments):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below", line_width=0,
            annotation_text=f"Segment {i+1}"
        )

    fig.update_layout(
        title="Audio Analysis with Detected Segments",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(x=0.01, y=0.99),
        height=400
    )
    
    return fig

def plot_rms_distribution(results_df, threshold=None):
    """Plot the distribution of RMS values with an optional threshold line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df['max_rms'], bins=50, alpha=0.7)
    
    if threshold is not None:
        ax.axvline(x=threshold, color='r', linestyle='--', label='Silence Threshold')
    elif 'is_silent' in results_df.columns:
        # Use the maximum RMS of silent files as the effective threshold
        silent_max = results_df[results_df['is_silent']]['max_rms'].max()
        ax.axvline(x=silent_max, color='r', linestyle='--', label='Silence Threshold')
    
    ax.set_xlabel('Maximum RMS Energy')
    ax.set_ylabel('Count')
    ax.legend()
    
    return fig 