import streamlit as st
import pandas as pd
import time
from scrapper import scrap
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import time
import requests
import os

st.sidebar.title('User Login')
Username = st.sidebar.text_input("Username:")
Password = st.sidebar.text_input("Password:", type="password")
# if not Username and Password:
#     st.error("Fill in this field")
Login = st.sidebar.button("Login")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.title("Blog Scribe!")
colsa, colsb = st.columns([1, 1])
with colsa:

    lottie_url_icon = "https://assets9.lottiefiles.com/private_files/lf30_F3v2Nj.json"
    lottie_icon = load_lottieurl(lottie_url_icon)
    st_lottie(lottie_icon, key='Hey')
# TO SELECT THE NUMBER OF BEAMS
with colsb:
    with st.expander("Click to set number of beams..."):
        st.markdown("Set no. of beams")
        slider_value = st.slider("", 1, 6)

    # BUTTON
    button_clicked = st.button("Click to view the entire data")

    st.markdown("---")
df = scrap()
option = st.selectbox(
    'Select the stock:',
    ('', 'TSLA', 'NIO'))
filtered_df = df[df['Stocks'] == option]
if option:
    # pd.set_option('display.max_colwidth', None)
    st.table(filtered_df.style.set_table_attributes(
        "style='width: 100%;'"))

if button_clicked:
    st.table(df.style.set_table_attributes("style='width: 100%;'"))
