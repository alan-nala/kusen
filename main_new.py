import streamlit as st
import whisper
import io
import tempfile


st.set_page_config(
    page_title="Kusen: transcriptor de audio a texto",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource
def transcribe_audio(file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
        try:
            temp_file.write(file.getvalue())
        except AttributeError:
            temp_file.write(file)
        temp_file.flush()
    model = whisper.load_model(model)
    transcript = model.transcribe(temp_file.name)
    return transcript


st.title("Kusen")
st.subheader("Ku: la boca 👄 ; Sen: la enseñanza 💫.")
st.write("Esta app transcribe audio a texto utilizando la librería OpenAI Whisper. Fue desarrollada para transcribir las enseñanzas dadas por el [Maestro Zen Soko Pierre Leroux](https://www.sokozen.org/)")
st.divider()
option = st.selectbox('Seleccione un modelo para la transcripción. (Se recomienda small)', ('small', 'base', 'tiny',''))
uploaded_file = st.file_uploader("Suba un archivo de audio", type=["wav", "mp3", "m4a", "ogg"])
# if st.button("Use test file"):
#     f = open("test.mp3", "rb")
#     uploaded_file = f.read()
#     f.close()
st.divider()
if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner("Transcribiendo el audio..."):
        transcript = transcribe_audio(uploaded_file, option)
    st.text_area("Transcripción:",transcript["text"], height=250)
else:
    st.write("Por favor suba un archivo de audio para transcribir.")