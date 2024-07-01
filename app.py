import base64
import streamlit as st
import librosa
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# Load the trained model
model = joblib.load('svm_best_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.set_page_config(page_title='DeepVoiceGuard',initial_sidebar_state='collapsed',layout='wide')

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_img_as_base64("bg_img.png")

# Custom styling
st.markdown(
f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

[data-testid='stFileUploader'] {{
    width: max-content;
}}
[data-testid='stFileUploader'] section {{
    padding: 0;
    float: left;
}}
[data-testid='stFileUploader'] section > input + div {{
    display: none;
}}
[data-testid='stFileUploader'] section + div {{
    float: right;
    padding-top: 0;
}}

.block-container {{
    padding-top: 5rem;
    padding-bottom: 0rem;
    padding-left: 5rem;
    padding-right: 5rem;
}}
</style>
""",
unsafe_allow_html=True
)

# Function to preprocess audio
def preprocess_audio(file_path, target_sr=22050, max_length=10):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
        max_len_samples = target_sr * max_length
        if len(y) > max_len_samples:
            y = y[:max_len_samples]
        else:
            y = np.pad(y, (0, max_len_samples - len(y)))
        return y, target_sr
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None, None

# Function for extracting MFCC, Chroma, and ZCR features
def extract_features(y, sr, duration=3, n_mfcc=13):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        features = np.hstack((mfccs_mean, chroma_mean, zcr_mean))
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None
    
# Sidebar navigation
with st.sidebar:
    selection = option_menu(
        menu_title = 'Navigation',
        options = ['About Deepfakes', 'About DeepVoiceGuard', 'Try DeepVoiceGuard'],
        default_index = 0,
    )

# About Deepfakes content
if selection == 'About Deepfakes':
    st.title('About Deepfakes and Deepfake Audios')
    st.markdown("""
    Deepfakes are synthetic media, such as videos, images, or audio, that have been manipulated using deep learning techniques. 
    In the context of audio, deepfake audios are artificially generated audio clips that mimic the voice of a person, often 
    created to deceive listeners into believing they are hearing a real person speak.
    
    The technology behind deepfakes has advanced rapidly, posing significant challenges for authenticity verification and 
    trustworthiness in media content.
    """)
    st.header("How are deepfake audios made ?",divider='rainbow')
    st.markdown("""
    Audio deepfakes are typically created using deep learning algorithms such as Generative Adversarial Networks (GANs) or Variational 
    Autoencoders (VAEs). These models analyze and synthesize speech patterns, intonations, and cadences from large datasets of recorded 
    speech. By training on these datasets, they can generate new audio clips that closely mimic the voice and speech characteristics of a 
    specific individual, even capturing nuances like accents and emotional inflections.
    """)

# About DeepVoiceGuard content
elif selection == 'About DeepVoiceGuard':
    st.title('About DeepVoiceGuard')
    st.markdown("""
    DeepVoiceGuard is an advanced AI-powered tool designed to detect deepfake audios with a 
    high degree of accuracy. In an era where synthetic media can blur the lines between reality 
    and fabrication, DeepVoiceGuard stands as a bulwark against the proliferation of deceptive audio content.
    
    **Technology Behind DeepVoiceGuard**:
    
    DeepVoiceGuard leverages state-of-the-art machine learning techniques, specifically a Support Vector Machine (SVM) 
    model trained on meticulously extracted audio features. These features include Mel-frequency cepstral coefficients (MFCCs), 
    Chroma features, and Zero Crossing Rate (ZCR), which collectively capture the unique nuances of human speech.
    
    **Predictive Power**:
    
    Through rigorous training on diverse datasets containing both bonafide and deepfake audios, DeepVoiceGuard 
    achieves an impressive accuracy rate of 96%. This robust predictive capability enables it to discern between 
    genuine audio recordings and artificially manipulated ones.
    
    **User Experience**:
    
    Users can seamlessly upload audio files to DeepVoiceGuard via a user-friendly interface. Upon upload, the 
    tool swiftly processes the audio, extracting pertinent features and providing real-time predictions regarding 
    its authenticity. Users can also listen to the uploaded audio directly within the app interface, enhancing 
    the understanding and verification process.
    """)

# Try DeepVoiceGuard content
elif selection == 'Try DeepVoiceGuard':
    st.title('Try DeepVoiceGuard')
    st.markdown("""
    In this section, you can try out DeepVoiceGuard. Upload an audio file, and DeepVoiceGuard will predict whether the audio 
    is bonafide (genuine) or spoofed (deepfake).
    """)



    
    st.write(np.zeros(1))



    
    # File upload
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Preprocess audio
        y, sr = preprocess_audio("temp_audio.wav")
        
        if y is not None:
            # Play the uploaded audio file
            st.audio("temp_audio.wav", format="audio/wav")

            # Extract features
            features = extract_features(y, sr)
            
            if features is not None:
                # Standardize features
                features = features.reshape(1, -1)
                features = scaler.transform(features)
                
                # Predict class
                prediction = model.predict(features)
                predicted_class = label_encoder.inverse_transform(prediction)
                
                # Display prediction
                st.success(f"Predicted class: {predicted_class[0]}")

                # Explanation of prediction
                if predicted_class[0] == 'bona-fide':
                    st.info("Bonafide means the audio is genuine and not altered or spoofed.")
                elif predicted_class[0] == 'spoof':
                    st.info("Spoof means the audio is artificially created or altered, possibly to deceive listeners.")
                
            else:
                st.error("Failed to extract features from the audio file.")
        else:
            st.error("Failed to preprocess the audio file.")

# Display sidebar and main content
st.sidebar.markdown("---")
st.sidebar.markdown("Explore the sections about Deepfakes and DeepVoiceGuard.")
