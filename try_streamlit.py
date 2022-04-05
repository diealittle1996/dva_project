import streamlit as st
import pandas as pd
import os


st.markdown("# Self Exploratory Visualization on the World of Paintings")
st.markdown("Explore the dataset to know more about artistic heritage")
st.markdown('''
Art is one of the world's great reservoirs of cultural heritage, and the rise of digitized open source art collections has made the sharing of this resource easier. The Metropolitan Museum, and  the Chicago Museum, and even Wikiart provide open source digitized versions of some of the world’s greatest artistic heritage.\n
Digitized collections further allow cross-cultural interaction. Newly available datasets offer new resources and potential in art history, art appreciation, design, and more.\n
Now, we invite you to delve into the world of art, find similar artworks, and form your own path of exploration adn inspiration.
''')


st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")


@st.cache(persist=True, show_spinner=True)

def load_data(nrows):
    df = pd.read_csv("MetObjects_subset.csv", nrows = nrows)
    return df



data_load_state = st.text('Loading dataset...')
df = load_data(100000)
data_load_state.text('Loading dataset...Completed!')


st.title('Quick  Explore')
st.sidebar.subheader(' Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")
if st.sidebar.checkbox('Dataset Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())
if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    st.write(df.columns.to_list())
if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())
