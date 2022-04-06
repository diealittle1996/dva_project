import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import db_dtypes
from PIL import Image
from image_matching import get_similar_art, get_image, new_image_as_df, extract_features_VGG
from sewar.full_ref import mse, rmse, uqi, scc, msssim, vifp


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


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
    query = f"SELECT * FROM `cse6242-343901.metobjects.table1` LIMIT {nrows}"
    df = client.query(query).to_dataframe()
    return df

data_load_state = st.text('Loading dataset...')
df = load_data(1000)
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
    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.write(df_types.astype(str))
    df['isHighlight'] = df['isHighlight'].apply(int)
    st.write(df.describe())
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())
    
ef_vgg = np.load("VGG_features.npy", encoding='bytes')
st.write(ef_vgg.shape)

df = pd.read_csv("cleaned_data_2.csv")
ef_vgg = np.load('VGG_features.npy')

TEST_IMAGE_ID = 37961
test_df_1 = new_image_as_df(TEST_IMAGE_ID, df)
ef_test_vgg = extract_features_VGG(test_df_1)

mylist, ordered_indices = get_similar_art(ef_vgg, ef_test_vgg, id_num=37961, df=df, distance="rmse")

new_art = get_image(TEST_IMAGE_ID)
similarity_vgg_euclidean = {"id":[],'mse': [], 'rmse': [],"scc":[],"uqi":[],"msssim":[],"vifp":[]}
for i in range(5):
    similar_art = get_image(mylist[i])
    similarity_vgg_euclidean["id"].append(mylist[i])
    similarity_vgg_euclidean["mse"].append(mse(new_art,similar_art))
    similarity_vgg_euclidean["rmse"].append(rmse(new_art,similar_art))
    similarity_vgg_euclidean["scc"].append(scc(new_art,similar_art))
    similarity_vgg_euclidean["uqi"].append(uqi(new_art,similar_art))
    similarity_vgg_euclidean["msssim"].append(msssim(new_art,similar_art).astype('float32'))
    similarity_vgg_euclidean["vifp"].append(vifp(new_art,similar_art))
similarity_vgg_euclidean_df = pd.DataFrame(similarity_vgg_euclidean)
st.write("avg mse: ", np.mean(similarity_vgg_euclidean_df["mse"]))
st.write(similarity_vgg_euclidean_df)


