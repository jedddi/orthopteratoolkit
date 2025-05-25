# 🦗 Orthoptera Analysis Toolkit

A comprehensive toolkit for analyzing, segmenting, and processing orthoptera (crickets, grasshoppers, etc.) audio recordings. 
(Note: Although it was made for Orthoptera Calls, it is not limited to other types of audio data)

## Overview

The Orthoptera Analysis Suite is an integrated application that combines multiple tools for working with orthoptera call recordings. It provides researchers and enthusiasts with a user-friendly interface to explore, analyze, segment, and process audio data from orthoptera species.

## Features

### 🔍 Dataset Browser
- Browse your dataset organized by species
- Visualize waveforms and spectrograms
- Listen to original and normalized audio
- Compare different normalization methods

### ✂️ Call Segmentation
- Detect and extract individual call segments from longer recordings
- Two detection methods:
  - Energy threshold-based detection
  - Peak detection for more rhythmic calls
- Preview and validate detected segments
- Save valid segments for further analysis

### 🔄 Batch Processing
- Process multiple files at once
- Extract call segments across your entire dataset
- Selectively choose which segments to save
- Customizable processing parameters per batch
- Save processing settings for reproducibility

### 🔇 Silent Audio Detector
- Scan your dataset for silent or nearly-silent audio files
- View statistics about silent files by species
- Visualize and listen to potentially problematic files
- Batch remove or move silent files

### ⏱️ Duration Normalizer
- Standardize the duration of audio files
- Add silence padding to shorter files
- Analyze duration distribution in your dataset
- Create uniform datasets for machine learning

## Installation

### Prerequisites
- Python 3.7+
- pip

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- librosa
- numpy
- pandas
- matplotlib
- plotly
- scipy
- pydub
- pillow

## Usage

### Running the Application
```bash
cd orthoptera_analysis_suite
streamlit run app.py
```

### Project Structure
```
orthoptera_analysis_suite/
├── app.py                 # Main application file
├── assets/                # Static assets (images, etc.)
├── utils/                 # Utility functions
│   ├── audio_processing.py  # Audio processing utilities
│   └── visualization.py     # Visualization utilities
└── modules/               # Feature modules
    ├── dataset_browser.py   # Dataset browsing module
    ├── call_segmentation.py # Call segmentation module
    ├── batch_processor.py   # Batch processing module
    ├── silence_detector.py  # Silent audio detector module
    └── duration_normalizer.py # Duration normalizer module
```

## Getting Started

1. Launch the application using the command above
2. Set your dataset path in the Settings page
3. Configure processing parameters to match your audio characteristics
4. Navigate to the specific tool you need using the sidebar
5. Process your data and save the results

## Dataset Format

The application expects your dataset to be organized in the following structure:
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

Where each species has its own folder containing audio recordings.

## Advanced Usage

### Audio Normalization
The application provides multiple methods for normalizing audio:
- **Min-Max**: Scales audio to the range [-1, 1]
- **Z-Score**: Standardizes audio to have zero mean and unit variance
- **Peak**: Scales audio to have a peak amplitude of 0.9
- **None**: Uses original audio without normalization

### Call Detection Methods
Two methods are available for detecting call segments:
- **Energy Threshold**: Identifies regions where audio energy exceeds a threshold
- **Peak Detection**: Identifies significant peaks in the audio energy envelope

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for orthoptera bioacoustics research
- Built with Streamlit, Librosa, and other open-source tools 
