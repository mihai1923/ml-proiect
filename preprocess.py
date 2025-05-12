import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
# functie pentru convertirea imaginii in numpy array
def preprocess_image_pil(image_path, target_size):
    img = Image.open(image_path)
    img_gray = img.convert('L') # convertim in imagine gray-scale
    img_resized_pil = img_gray.resize(target_size)  # redimensionam toate imaginile la 128x128
    img_array = np.array(img_resized_pil)  # convertim intr-un array numpy
    return img_array

# functie pentru extragerea caracteristicilor
def extract_image_features(image_gray):
    features = {}
    flat_pixels = image_gray.flatten()
    # incercam cat pe mult sa ignoram background-ul negru
    # care strica valorile caracteristicilor folosind threshold_otsu
    uniq_vals = np.unique(image_gray)
    thresh = threshold_otsu(image_gray)
    brain_pixels = flat_pixels[flat_pixels > thresh]
    features['mean_intensity'] = np.mean(brain_pixels)
    features['std_intensity'] = np.std(brain_pixels)
    features['min_intensity'] = np.min(brain_pixels)
    features['max_intensity'] = np.max(flat_pixels)
    features['intensity_range'] = features['max_intensity'] - features['min_intensity']
    features['median_intensity'] = np.median(flat_pixels)
    features['q1_intensity'] = np.percentile(flat_pixels, 25)
    features['q3_intensity'] = np.percentile(flat_pixels, 75)
    return features

directory = 'All_Tumor_Labeled'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = (256, 256)
output_csv = 'brain_tumor_features.csv'
all_features_list = []
print(f"Inceput extragere caracteristici...")
# iteram prin toate tipurile de tumori
for category_idx, category in enumerate(categories):
    category_path = os.path.join(directory, category)
    print(f"Procesare categorie: {category}...")
    # luam imaginile din tipul 'category' de tumori
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith('.jpg')]
    # iteram prin fiecare imagine
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(category_path, image_name)
        # transforma imaginea intr-un numpy array ca sa putem lucra cu array-ul creat
        preprocessed_img_array = preprocess_image_pil(image_path, img_size)
        # extrage caracteristicile din array-ul creat
        img_features = extract_image_features(preprocessed_img_array)
        # adaugam tipul tumorii printre caracteristici
        img_features['tumor_type'] = category
        # punem toate caracteristicile in lista
        all_features_list.append(img_features)
        # afisam la fiecare 200 de iteratii sa putem vedea progresul
        if (i + 1) % 200 == 0:
            print(f"  Procesat {i + 1}/{len(image_files)} imagini in {category}...")
    print(f"Finalizat procesarea a {len(image_files)} imagini pentru categoria: {category}.")
features_df = pd.DataFrame(all_features_list)
# odata terminat afisam un mesaj de finalizare extragere
# si afisam primele 5 randuri + caracteristicile dataframe-ului
print(f"Extragerea caracteristicilor a {len(all_features_list)} finalizata.")
print(features_df.head())
print("Informatii DataFrame:")
features_df.info()
# salvam datele intr-un csv
features_df.to_csv(output_csv, index=False)