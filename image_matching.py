import streamlit as st
import numpy as np
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.metrics import pairwise_distances
from sewar.full_ref import rmse
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import io
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

def download_blob_into_memory(bucket_name, blob_name):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    contents = blob.download_as_string()
    return contents

def display_test_image(test_image):
    st.image(Image.open("dva_paintings/"+test_image))

def get_image(test_image, show=0):
    try:
        int(test_image.replace('TP_',''))
        im = download_blob_into_memory('dva_paintings', f'{test_image}.jpg')
    except:
        try:
            im = download_blob_into_memory('dva_paintings', f'{test_image}')
        except:
            im = Image.open("dva_paintings/"+test_image)
            myImage = np.array(im)
            return np.resize(myImage,(100,100))
    fp = io.BytesIO(im)
    myImage = mpimg.imread(fp, format='jpeg')
    if show:
        st.image(myImage)
    else:
        return np.resize(myImage,(100,100))

def getCosineSimilarity(A, B):
    cos_similarity = np.dot(A,B.T) / (np.linalg.norm(A)*np.linalg.norm(B))
    return cos_similarity[0][0]

def eliminate_dup(test, indices, feature_df):
    ids = [str(index_to_id(idx, feature_df)) for idx in indices]
    myID = '.'.join(test.split(".")[:-1])
    if myID in ids:
        new_indices = np.delete(indices, ids.index(myID))
    else:
        new_indices = indices
    return new_indices

def get_similar_art(extracted_features, new_art_ef, test, feature_df, df, count=5,distance = "euclidean"):
    if distance == "euclidean":
        dist = pairwise_distances(extracted_features, new_art_ef).T[0]
        indices = np.argsort(dist)[0:count]
        indices = eliminate_dup(test, indices, feature_df)
        if len(indices) < 5:
            indices = np.append(indices, np.argsort(dist)[count])
        pdists  = np.sort(dist)[0:count]
    elif distance == "cosine":
        dist = []
        for feature in extracted_features:
            dist.append(getCosineSimilarity(feature.reshape(1,extracted_features.shape[1]), new_art_ef))
        indices = np.argsort(dist)[0:count]
        indices = eliminate_dup(test, indices, feature_df)
        if len(indices) < 5:
            indices = np.append(indices, np.argsort(dist)[count])
        pdists  = np.sort(dist)[0:count]
    elif distance == "rmse":
        dist = []
        for feature in extracted_features:
            dist.append(rmse(feature.reshape(1,extracted_features.shape[1]), new_art_ef))
        indices = np.argsort(dist)[0:count]
        indices = eliminate_dup(test, indices, feature_df)
        if len(indices) < 5:
            indices = np.append(indices, np.argsort(dist)[count])
        pdists  = np.sort(dist)[0:count]

    min_elements =  np.array(dist)[indices]

    min_elements_order = np.argsort(min_elements)
    ordered_indices = indices[min_elements_order]

    mylist = []
    i=-1
    for index in ordered_indices:
        i+=1
        objectID = index_to_id(index, feature_df)
        mylist.append(objectID)
    return mylist, ordered_indices

def extract_features_VGG(dataframe, img_width=224, img_height=224, batch_size=64, save=False):
    num_samples = len(dataframe)
    Itemcodes = []
    datagen = ImageDataGenerator(rescale=1. / 255,)
    model = VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_dataframe(
        dataframe,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=False,
        class_mode=None)

    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/")+1):i.find(".")])

    extracted_features = model.predict(generator, num_samples // batch_size)
    extracted_features = extracted_features.reshape((num_samples, 25088))

    if save==True:
        np.save(open('VGG_features.npy', 'wb'), extracted_features)
    return extracted_features

def new_image_as_df(test_image):
    if isinstance(test_image, str):
        new_image = 'dva_paintings/' + test_image
    if isinstance(test_image, int):
        new_image = 'dva_paintings/' + str(test_image) + '.jpg'
    new_df = pd.DataFrame()
    new_df['filename'] = [new_image]
    return new_df

def index_to_id(index, df):
    return df['objectID'][index]
