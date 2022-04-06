import streamlit as st
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import pairwise_distances
from sewar.full_ref import rmse
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import io
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/tianhuazhu/Downloads/key3.json"

def download_blob_into_memory(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    contents = blob.download_as_string()
    return contents

def get_image(id_num, show=0):
    im = download_blob_into_memory('dva_paintings', f'{id_num}.jpg')
    fp = io.BytesIO(im)
    myImage = mpimg.imread(fp, format='jpeg')
    if show:
        # plt.imshow(myImage)
        # plt.show()
        st.image(myImage)
    else:
        return np.resize(myImage,(100,100))

def getCosineSimilarity(A, B):
    cos_similarity = np.dot(A,B.T) / (np.linalg.norm(A)*np.linalg.norm(B))
    return cos_similarity[0][0]

def get_similar_art(extracted_features, new_art_ef, id_num, df, count=5,distance = "euclidean"):
    if distance == "euclidean":
        dist = pairwise_distances(extracted_features, new_art_ef).T[0]
        indices = np.argsort(dist)[0:count]
        pdists  = np.sort(dist)[0:count]
    elif distance == "cosine":
        dist = []
        for feature in extracted_features:
            dist.append(getCosineSimilarity(feature.reshape(1,extracted_features.shape[1]), new_art_ef))
        indices = np.argsort(dist)[0:count]
        pdists  = np.sort(dist)[0:count]

    elif distance == "rmse":
        dist = []
        for feature in extracted_features:
            dist.append(rmse(feature.reshape(1,extracted_features.shape[1]), new_art_ef))
        indices = np.argsort(dist)[0:count]
        pdists  = np.sort(dist)[0:count]

    min_elements =  np.array(dist)[indices]

    min_elements_order = np.argsort(min_elements)
    ordered_indices = indices[min_elements_order]

    st.write("="*20 + "input product image" + "="*20)
    get_image(id_num, 1)
    mylist = []

    st.write("\n","="*20 + "Similar Images" + "="*20)
    i=-1
    for index in ordered_indices:
        i+=1
        objectID = index_to_id(index, df)
        mylist.append(objectID)
        get_image(objectID, 1)
        st.write('Distance from input image:' + str(pdists[i]))
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

def new_image_as_df(new_image, df):
    new_image = 'dva_paintings/' + str(new_image) + '.jpg'
    new_df = pd.DataFrame()
    new_df['filename'] = [new_image]
    return new_df

def index_to_id(index, df):
    return df['objectID'][index]