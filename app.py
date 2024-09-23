import streamlit as st
import whisper
from tempfile import NamedTemporaryFile
st.set_page_config(page_title="whisper ASR", page_icon="musical_note",layout="wide")
#CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
bg = """
        <style> [data-testid="stAppViewContainer"]{
        background: rgb(6,36,39);
        }
        </style>
        """
st.markdown(bg, unsafe_allow_html=True)

st.title("🔉 Automatic Audio and Video transciption app using Whisper by OpenAI 🔉")
st.info('✨ Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV and video format - MP4')
#upload audio file with streamlit
audio_file = st.file_uploader("Upload Audio or video", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

if audio_file is not None:
            #importing model -- base(74M pararameter)
            model = whisper.load_model("base")
            st.info("Whisper model loaded")

            #playing audio file
            st.header("Play audio file:")
            st.audio(audio_file)

            #genrating transcript
            if st.button("Generate Trasnscript"):
                        with st.spinner(f"Processing Audio ... 💫"):
                                if audio_file is not None:
                                    with NamedTemporaryFile(suffix="mp3") as temp:
                                        temp.write(audio_file.getvalue())
                                        temp.seek(0)
                                        st.success("Transcribing Audio/video")
                                        transcription = model.transcribe(temp.name,fp16=False)
                                        st.success("Transcription Complete")
                                        st.markdown(transcription["text"])

                                        # Save the transcript to a text file
                                        with open("transcript.txt", "w") as f:
                                            f.write(transcription["text"])

                                        # Provide a download button for the transcript
                                        st.download_button("Download Transcript", transcription["text"])
                                else:
                                    st.error("⚠ Please upload a audio file")
