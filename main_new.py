import whisper 
import os
import numpy as np
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio
import streamlit as st

from tqdm.notebook import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title('Carga de archivos de audio')

#Cargar audio
uploaded_file = st.file_uploader("Carga un archivo de audio", type='wav')
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav')
    with open("/tmp/audio.wav", "wb") as f:
        f.write(audio_bytes)
    st.warning("Waiting for file to be uploaded and processed...")
    st.write('File name: `%s`' % uploaded_file)
    data = whisper.load_audio("/tmp/audio.wav")
    
    # We can load one of the models here, the base model requires fewest amount of memory
    model = whisper.load_model("base") # change base small medium large

    # This is the high level way of transcribing your audio into text.
    result = model.transcribe(data)
    
    transcripcion = result["text"]

    # Result is a dictionary containing "text" and "segments" and "language"
    st.write(result["text"])

    # There is also a lower level way that allows you to tweak your audio sample
    audio = whisper.pad_or_trim(torch.from_numpy(data).float())
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    # This allows us to check for other potential candidates, 
    # for example if we want to do some debugging or calculating some metrics
    st.write(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    st.write(result.text)
    
    # There is also a lower level way that allows you to tweak your audio sample
    # audio = whisper.pad_or_trim(torch.from_numpy(data).float())
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # _, probs = model.detect_language(mel)

    # This allows us to check for other potential candidates, 
    # for example if we want to do some debugging or calculating some metrics
    # st.write(f"Detected language: {max(probs, key=probs.get)}")
    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)

    #obtener la ruta del archivo cargado
    ruta = uploaded_file.name
    f = open(ruta,"w+")
    text_file = open(ruta,"w")

    #write string to file
    text_file.write(transcripcion)

    #close file
    text_file.close()

    #Descargar el archivo
    st.download_button(
        label="Descargar transcripción",
        data=text_file,
        file_name= ruta+'.txt',
        mime='text/plain')
else: 
    st.warning("No file uploaded")
