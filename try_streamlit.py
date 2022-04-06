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
    query2 = "SELECT * FROM `cse6242-343901.metobjects.table1`"
    df = client.query(query).to_dataframe()
    full_df = client.query(query2).to_dataframe()
    return df, full_df

data_load_state = st.text('Loading dataset...')
df, full_df = load_data(1000)
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
 
st.subheader("Provide Your Image to Find Visually Similar Ones:")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                  "filesize":image_file.size}
    st.write(file_details)
    with open(os.path.join("dva_paintings",image_file.name),"wb") as f:
        f.write((image_file).getbuffer())
        st.write(os.path.join("dva_paintings",image_file.name))
        st.success("New Image Received")
    button = st.button('Generate recommendations')
    if button:
        dta = pd.read_csv("cleaned_data_2.csv")
        ef_vgg = np.load('VGG_features.npy')

        TEST_IMAGE = image_file.name
        test_df_1 = new_image_as_df(TEST_IMAGE)
        ef_test_vgg = extract_features_VGG(test_df_1)

        mylist, ordered_indices = get_similar_art(ef_vgg, ef_test_vgg, test=TEST_IMAGE, feature_df=dta, df=full_df.astype(str), distance="rmse")

        new_art = get_image(TEST_IMAGE)
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
