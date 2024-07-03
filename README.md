# DeepVoiceGuard

DeepVoiceGuard is an AI-powered tool designed to detect deepfake audios. It utilizes a Support Vector Machine (SVM) model trained on extracted audio features to predict whether an audio clip is bonafide (genuine) or spoofed (deepfake). This project demonstrates the use of machine learning in enhancing trust and authenticity in digital audio content.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [About Deepfakes](#about-deepfakes)
- [About DeepVoiceGuard](#about-deepvoiceguard)
- [Try DeepVoiceGuard](#try-deepvoiceguard)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Deepfakes are synthetic media, such as videos, images, or audio, that have been manipulated using deep learning techniques. In the context of audio, deepfake audios are artificially generated audio clips that mimic the voice of a person, often created to deceive listeners into believing they are hearing a real person speak. The technology behind deepfakes has advanced rapidly, posing significant challenges for authenticity verification and trustworthiness in media content.

## Features

- Preprocess audio files for consistency in sample rate and length.
- Extract audio features including MFCC, Chroma, and Zero Crossing Rate.
- Predict whether an audio file is bonafide or spoofed using an SVM model.
- User-friendly interface with Streamlit for easy audio file uploads and predictions.

## About Deepfakes

Deepfake audios are typically created using deep learning algorithms such as Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs). These models analyze and synthesize speech patterns, intonations, and cadences from large datasets of recorded speech. By training on these datasets, they can generate new audio clips that closely mimic the voice and speech characteristics of a specific individual, even capturing nuances like accents and emotional inflections.

Deepfake audios can be differentiated from genuine audios through various techniques such as analyzing inconsistencies in speech patterns, detecting unnatural audio artifacts, and using machine learning models trained to identify features unique to deepfake audio.

## About DeepVoiceGuard

DeepVoiceGuard utilizes an SVM model trained on extracted audio features to determine if an audio clip is authentic or a spoof. This model achieves a 96% accuracy rate in identifying genuine versus deepfake audios.

The system extracts features such as MFCC, Chroma, and Zero Crossing Rate from audio files. These features are standardized before being input into the SVM model for prediction. DeepVoiceGuard's accuracy in distinguishing between bonafide and deepfake audios is 96%.

The primary goal of DeepVoiceGuard is to provide a robust defense against the spread of deepfake audio content, thereby enhancing trust and authenticity in digital media.
## Try DeepVoiceGuard

Try out DeepVoiceGuard [here](https://deepvoiceguard.streamlit.app/). Upload an audio file, and DeepVoiceGuard will predict whether the audio is bonafide (genuine) or spoofed (deepfake).

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/rheachainani/deepvoiceguard.git
   cd deepvoiceguard
2. **Create a virtual environment:**
   ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `.\venv\Scripts\activate`
3. **Install the required packages:**
   ```sh
    pip install -r requirements.txt

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

2. **Navigate to the web interface:**
   Open your web browser and go to `http://localhost:8501`.

3. **Use the interface:**
   - Upload an audio file.
   - View the prediction results (bonafide or spoofed).
   - Read information about deepfakes and the DeepVoiceGuard project.
