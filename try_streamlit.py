import streamlit as st
import pandas as pd
import numpy as np
import folium
from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import db_dtypes
from PIL import Image
from image_matching import get_similar_art, get_image, new_image_as_df, extract_features_VGG, display_test_image
from sewar.full_ref import mse, rmse, uqi, scc, msssim, vifp
from streamlit_folium import folium_static
import json
from pathlib import Path
import cv2 as cv


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

def download_image(_id):
    try:
        int(_id.replace('TP_',''))
        im = download_blob_into_memory('dva_paintings', f'{_id}.jpg')
    except:
        try:
            im = download_blob_into_memory('dva_paintings', f'{_id}')
        except:
            im = Image.open("dva_paintings/"+_id)
            myImage = np.array(im)
            return myImage
    fp = io.BytesIO(im)
    myImage = mpimg.imread(fp, format='jpeg')
    return myImage
    
def display_images(test_img, cap_fields, ids, df):

    st.subheader("=" * 10 + "  User image  " + "=" * 10)
    display_test_image(test_img)

    captions = {id: {field: df.loc[df.objectID == str(id)][field].values[0] for field in cap_fields} for id in ids}
    # st.write(captions)

    st.subheader("\n", "=" * 10 + f"  Top {num_similar_paintings} Similar Images  " + "=" * 10)

    # This part assumes all database images are stored locally in path/images folder.
    # st.write(imgs)

    num_imgs = len(ids)

    st.write("\n", "=" * 20 + f"  Top {num_imgs} Similar Images  " + "=" * 20)
    idx = 0
    for _ in range(num_imgs - 1):
        cols = st.columns(4)

        if idx < num_imgs:
            cols[0].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]])
        idx += 1

        if idx < num_imgs:
            cols[1].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]])
        idx += 1

        if idx < num_imgs:
            cols[2].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]])
        idx += 1
        if idx < num_imgs:
            cols[3].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]])
            idx = idx + 1
        else:
            break

# Callback functions to preserve button functionalities. Not working yet.
def disp_imgs_CB():
    st.session_state.active_page = "display"

def choropleth_CB():
    st.session_state.active_page = "choropleth"
    
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

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Home'

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
    new_line = " \n\n\n "
    info = [f"{key}: {file_details[key]}" for key in file_details.keys()]
    image_details = f"Image details:{new_line}{new_line.join(map(str, info))}"
    st.write(image_details)
    
    with open(os.path.join("dva_paintings",image_file.name),"wb") as f:
        f.write((image_file).getbuffer())
        st.success("New Image Received")
        
    user_input = st.text_input("How many similar images would you like to find?",
                               help="Try entering a number larger than 5.")
    if len(user_input) != 0:
        num_similar_paintings = int(user_input)

        # Load necessary info.
        processing = st.text("Processing...")
        data = pd.read_csv("cleaned_data_3.csv")
        ef_vgg = np.load('VGG_features.npy')

        TEST_IMAGE = image_file.name
        test_image_df = new_image_as_df(TEST_IMAGE)
        ef_test_vgg = extract_features_VGG(test_image_df)

        # Find similar paintings.
        similar_img_ids, ordered_indices = get_similar_art(ef_vgg,
                                                           ef_test_vgg,
                                                           test=TEST_IMAGE,
                                                           feature_df=data,
                                                           df=full_df.astype(str),
                                                           count=num_similar_paintings,
                                                           distance="rmse")

        # Create dataframe as input for choropleth.
        # map_info_df made from {"Region": [unique contry names], "Counts": [count_per_country]}
        countries = [data.loc[data.objectID == str(id)]["Country"].values[0] for id in similar_img_ids]
        countries_unique = np.unique(countries)
        counts = [countries.count(x) for x in countries_unique]
        map_info = {"Region": countries_unique,
                    "Counts": counts}
        map_info_df = pd.DataFrame(map_info)

        processing.text("Processing... Completed!")

        # Display buttons.
        col1, col2 = st.columns(2)

        gen_rec_button = col1.button(f'Display your {num_similar_paintings} recommendations', on_click=disp_imgs_CB)
        gen_choropleth_button = col2.button("Generate a choropleth", on_click=choropleth_CB)

        # gen_rec_button = st.button(f'Display your {num_similar_paintings} recommendations', key=1)
        # gen_choropleth_button = st.button("Generate a choropleth", key=2)

        if st.session_state.active_page == "display":

            fields = ["Title", "ArtistName", "TimePeriod", "Country"]

            display_images(TEST_IMAGE, fields, similar_img_ids, data)

        if st.session_state.active_page == "choropleth":
            load_choropleth = st.empty()
            load_choropleth.markdown("Loading choropleth...")

            # Folium documentation: https://python-visualization.github.io/folium/modules.html
            # https://python-visualization.github.io/folium/quickstart.html#Choropleth-maps

            # Initialize map centered on Beijing.
            map = folium.Map(location=[35.8617, 104.1954], zoom_start=3)

            folium.Choropleth(
                geo_data=f"countries.geojson",
                name="choropleth",
                data=map_info_df,
                columns=["Region", "Counts"],
                key_on="feature.properties.ADMIN",
                fill_color="YlGn",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Count of similar paintings by country"
            ).add_to(map)

            # Displays map.
            folium_static(map)
            load_choropleth.empty()
            st.caption("This map displays the country of origin for similar artworks. Hover over the country to see the corresponding artworks.\n You may need to zoom out to see all the relevant countries.")
            
#         new_art = get_image(TEST_IMAGE)
#         similarity_vgg_euclidean = {"id":[],'mse': [], 'rmse': [],"scc":[],"uqi":[],"msssim":[],"vifp":[]}
#         for i in range(5):
#             similar_art = get_image(mylist[i])
#             similarity_vgg_euclidean["id"].append(mylist[i])
#             similarity_vgg_euclidean["mse"].append(mse(new_art,similar_art))
#             similarity_vgg_euclidean["rmse"].append(rmse(new_art,similar_art))
#             similarity_vgg_euclidean["scc"].append(scc(new_art,similar_art))
#             similarity_vgg_euclidean["uqi"].append(uqi(new_art,similar_art))
#             similarity_vgg_euclidean["msssim"].append(msssim(new_art,similar_art).astype('float32'))
#             similarity_vgg_euclidean["vifp"].append(vifp(new_art,similar_art))
#         similarity_vgg_euclidean_df = pd.DataFrame(similarity_vgg_euclidean)
#         st.write("avg mse: ", np.mean(similarity_vgg_euclidean_df["mse"]))
#         st.write(similarity_vgg_euclidean_df)
