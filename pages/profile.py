import streamlit as st
from PIL import Image
import base64

image = Image.open('Assets/Arcane.jpeg')
st.image(image)

st.write("## For more information visit: [Github](https://github.com/CoulsonZhang/6893BigData)")
st.write("## Team: Song Wen, Zheyu Zhang")

