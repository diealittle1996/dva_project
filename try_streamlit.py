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
from image_matching import get_similar_art, get_image, new_image_as_df, extract_features_VGG, display_test_image, download_image
from sewar.full_ref import mse, rmse, uqi, scc, msssim, vifp
from streamlit_folium import folium_static
import json
import geojson
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# client = bigquery.Client(credentials=credentials)

st.session_state.update(st.session_state)

st.markdown("# Self Exploratory Visualization on the World of Paintings")
st.markdown("Explore the dataset to know more about artistic heritage")
st.markdown('''
Art is one of the world's great reservoirs of cultural heritage, and the rise of digitized open source art collections has made the sharing of this resource easier. The Metropolitan Museum, and  the Chicago Museum, and even Wikiart provide open source digitized versions of some of the world’s greatest artistic heritage.\n
Digitized collections further allow cross-cultural interaction. Newly available datasets offer new resources and potential in art history, art appreciation, design, and more.\n
Now, we invite you to delve into the world of art, find similar artworks, and form your own path of exploration adn inspiration.
''')


st.sidebar.markdown("## Geographic Filter")
st.sidebar.markdown("Use this panel to filter for your countries of interest.")

# @st.cache(persist=True, show_spinner=True, suppress_st_warning=True)   
def display_images(test_img, cap_fields, ids, df):

    st.subheader("=" * 10 + "  User image  " + "=" * 10)
    display_test_image(test_img)
    
    # A dictionary of caption fields pulled from df with image id as key value.
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
            cols[0].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]] + " (" + ', '.join([captions[ids[idx]][i] for i in fields[1:4]]) + ")")
            cols[0].write("Full image: "+ captions[ids[idx]][fields[4]])
        idx += 1

        if idx < num_imgs:
            cols[1].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]] + " (" + ', '.join([captions[ids[idx]][i] for i in fields[1:4]]) + ")")
            cols[1].write("Full image: "+ captions[ids[idx]][fields[4]])
        idx += 1

        if idx < num_imgs:
            cols[2].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]] + " (" + ', '.join([captions[ids[idx]][i] for i in fields[1:4]]) + ")")
            cols[2].write("Full image: "+ captions[ids[idx]][fields[4]])
        idx += 1
        
        if idx < num_imgs:
            cols[3].image(download_image(ids[idx]), width=150, caption=captions[ids[idx]][fields[0]] + " (" + ', '.join([captions[ids[idx]][i] for i in fields[1:4]]) + ")")
            cols[3].write("Full image: "+ captions[ids[idx]][fields[4]])
            idx += 1
        else:
            break

# def load_data(nrows):
#     query = f"SELECT * FROM `cse6242-343901.metobjects.table1` LIMIT {nrows}"
#     query2 = "SELECT * FROM `cse6242-343901.metobjects.table1`"
#     df = client.query(query).to_dataframe()
#     full_df = client.query(query2).to_dataframe()
#     return df, full_df

data_load_state = st.text('Loading dataset...')
data = pd.read_csv("cleaned_data.csv")
# df, full_df = load_data(1000)
data_load_state.text('Loading dataset...Completed!')
if st.checkbox("Display Full Dataset"):
    st.write(data)
    
#################### Build the sidebar ####################
filters = {}
filters['Country'] = {}
filters['Century'] = {}

# Get the lists of countries (shown alphabetically)
# This covers all high-level filters
geo = {
    "All Art": ['Austria', 'Belgium', 'Britain', 'China', 'Denmark', 'France', 'Germany', 'Greece', 'Hungary', 'India',
                'Ireland', 'Italy', 'Japan', 'Korea', 'Nepal', 'Netherlands', 'Norway', 'Russia', 'Spain', 'Sri Lanka',
                'Sweden', 'Switzerland', 'Thailand', 'Unknown'],
    "Asian Art Only": ['China', 'India', 'Japan', 'Korea', 'Nepal', 'Sri Lanka', 'Thailand'],
    "European Art Only": ['Austria', 'Belgium', 'Britain', 'Denmark', 'France', 'Ireland', 'Germany', 'Greece',
                     'Hungary', 'Italy', 'Netherlands', 'Norway', 'Russia', 'Spain', 'Sweden', 'Switzerland'
                     ]
}

# Option for user to select their filter
selected = st.sidebar.radio("Quick Filter", ['All Art', 'Asian Art Only', 'European Art Only'])
# Apply the filter to a list to be used later on
st.sidebar.write('Further Filter by individual countries')
countries = geo[selected]
for i in countries:
    filters['Country'][i] = st.sidebar.checkbox(i, value=True, key=i)
#################### End of sidebar ####################


st.subheader("Time Filter")
periods = ['Before 1000', "1000's", "1100's", "1200's", "1300's", "1400's","1500's",
           "1600's", "1700's", "1800's", "1900's", "2000's"]
choice = ["Filter by time slider", "Filter by checkbox"]
choice = st.selectbox("Choose how you would like to filter time", choice)

if choice == "Filter by time slider":
    century = st.select_slider("Choose a continuous time period of interest", options = periods, value=('Before 1000', "2000's"))
    begin, end = century
    idx1 = periods.index(begin)
    idx2 = periods.index(end)
    for i in range(idx1, idx2+1):
        filters['Century'][periods[i]] = True
    filters['Century']['Unavailable'] = True
else:
    for i in periods:
        filters['Century'][i] = st.checkbox(i, value=True, key=i)
    filters['Century']['Unavailable'] = st.checkbox('Unavailable', value=True, key="time unavailable")

def update_filter(filter_dict, df):
    # Input is a dictionary of True/False labels
    remove_list = []

    # get list of IDs to remove
    for bigkey in filter_dict:
        for key, value in filter_dict[bigkey].items():
            if value == False:
                remove = list(df[df[bigkey] == key]['objectID'].values)
                remove_list += remove
        remove_list += list(df[~df[bigkey].isin(list(filter_dict[bigkey].keys()))]['objectID'].values)

    # start with a full list
    full_list = df['objectID'].values
    filtered_list = [a for a in full_list if a not in remove_list]
    return filtered_list

filtered_ids = update_filter(filters, data)
df_filtered = data.loc[data['objectID'].isin(filtered_ids)]
st.subheader('Dataset Quick Look:')
st.write(df_filtered.head())
st.write(len(df_filtered), "data points left")

if len(df_filtered) < 10:
    st.markdown("#### Warning: too few data points!")
else:
    val = list(df_filtered['Century'].values)
    counts = []
    for u in periods+["Unavailable"]:
        counts.append(val.count(u))

    freq_df = df_filtered.groupby(['Century','Country'])['Century'].size().unstack('Country').fillna(0)
    new_index = periods+["Unavailable"]
    freq_df = freq_df.reindex(new_index)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 6)
    freq_df.plot(kind="bar", stacked=True, ax=ax, legend=False)
    ax.set_title("Period Distribution in the Selected Countries (Interactive)")
    ax.set_xlabel("Century")
    ax.set_ylabel("Frequency")
    ax.legend(bbox_to_anchor= (1.02, 1))
    ax.set_xticklabels(periods+["Unavailable"])

    for i, v in enumerate(counts):
        plt.text(i-0.12, v+5, str(v))

    st.pyplot(fig)
 

st.markdown("#### \**Please beware that filters will be applied to the image matching algorithm. Results may be compromised if too many filters are applied. \**")
st.subheader("Submit Your Image to Find Visually Similar Ones:")
image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
selected_countries = [k for k,v in filters['Country'].items() if v==True]
selected_periods = [k for k,v in filters['Century'].items() if v==True]

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
        
    if len(selected_countries) > 0:
            idx = list(data[~data['Country'].isin(selected_countries)].index)
    if len(selected_periods) > 0:
        idx += list(data[~data['Century'].isin(selected_periods)].index)
    idx = np.unique(idx)
    st.write(len(idx), "image ruled out!")
        
    user_input = st.text_input("How many similar images would you like to find? (Press enter to display results)",
                               help="Try entering a number larger than 5.")
    if len(user_input) != 0:
        try:
            int(user_input)
        except:
            st.write("Please enter an integer!")
        else:
            if (int(user_input) > (len(data)-len(idx))) | (int(user_input) < 0):
                st.write(f"Requested number out of data range! Please enter a positive number smaller than {len(data)-len(idx)}.")
            else:
                num_similar_paintings = int(user_input)

                # Load necessary info.
                with st.spinner("Processing..."):
                    ef_vgg = np.load('VGG_features.npy')
                    ef_vgg = np.delete(ef_vgg, list(set(idx)), 0)
                    data2 = data[~data.index.isin(idx)].reset_index(drop=True).copy()

                    TEST_IMAGE = image_file.name
                    test_image_df = new_image_as_df(TEST_IMAGE)
                    ef_test_vgg = extract_features_VGG(test_image_df)

                    # Find similar paintings.
                    similar_img_ids, ordered_indices = get_similar_art(ef_vgg,
                                                                       ef_test_vgg,
                                                                       test=TEST_IMAGE,
                                                                       feature_df=data2,
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

                st.success("Pre-processing completed!")

                # Display images.
                fields = ["Title", "ArtistName", "Country", "Century","primaryImage"]
                display_images(TEST_IMAGE, fields, similar_img_ids, data)

                load_choropleth = st.empty()
                load_choropleth.markdown("Loading choropleth...")

                # Folium documentation: https://python-visualization.github.io/folium/modules.html
                # https://python-visualization.github.io/folium/quickstart.html#Choropleth-maps

                # Initialize map centered on continent with highest frequency.
                max_count_country = map_info_df.loc[[map_info_df["Counts"].idxmax()]]["Region"].values[0]
                loc = [48.019, 66.923]

                if max_count_country in geo["Asian Art Only"]:
                    loc = [39.916, 116.383]

                if max_count_country in geo["European Art Only"]:
                    loc = [50.378, 14.970]

                map = folium.Map(location=loc, min_zoom=2, max_zoom=5, zoom_start=3)
                geo_file = f"countries.geojson"

                with open(geo_file, encoding="utf8") as f:
                    map_data = geojson.load(f)

                # Trim down map_data to include only countries in database.
                trimmed_map_data = {"type": "FeatureCollection",
                                    "features": []
                                    }

                nl = "<br />"
                for country in map_data["features"]:
                    if country["properties"]["ADMIN"] in geo["All Art"]:
                        trimmed_map_data["features"].append(country)
                        first_line = "Country name: {}".format(country['properties']['ADMIN'])
                        second_line = ["Number of recommended paintings: 0"]
                        trimmed_map_data["features"][-1]["properties"]['text'] = f"{first_line}{nl}{nl.join(second_line)}"

                # st.text(trimmed_map_data["features"][0]["properties"])

                # Build tooltip.
                tooltip_text = {}
                for i, country in enumerate(countries_unique):
                    first_line = "Country name: {}".format(countries_unique[i])
                    second_line = ["Number of recommended paintings: {}".format(counts[i])]
                    tooltip_text[country] = f"{first_line}{nl}{nl.join(second_line)}"

                # st.text(tooltip_text["China"])

                # Find the index in the JSON file for each country.
                index_in_json = {}
                for idx, country in enumerate(trimmed_map_data["features"]):
                    index_in_json[country['properties']['ADMIN']] = idx

                # st.write(index_in_json.keys())
                # st.write(tooltip_text.keys())

                for key in tooltip_text.keys():
                    trimmed_map_data["features"][index_in_json[key]]["properties"]['text'] = tooltip_text[key]

                # st.text(trimmed_map_data["features"][index_in_json["China"]]["properties"])

                choropleth = folium.Choropleth(geo_data=trimmed_map_data,
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
                choropleth.geojson.add_child(
                    folium.features.GeoJsonTooltip(['text'], labels=False)
                )
                folium_static(map)
                load_choropleth.empty()
                st.caption("This map displays the country of origin for similar artworks. You may need to zoom out to see all of the relevant countries.\n Note that only countries in our database are colored (non-white).")
     
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
