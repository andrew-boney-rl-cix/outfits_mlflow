#libraries
import pandas as pd

import dataiku
import os

from mlflow_utils import main, get_image_name
from my_model import CLIPComplete

#data
prods = dataiku.Dataset("ralph_lauren_full_feed").get_dataframe()
img_folder = "code_studio-versioned/pics/"
imgs = img_folder + pd.Series(os.listdir(img_folder))

pairs = (prods[["ProductImageURL", "lookProductImage1"]]
     .drop_duplicates()
     .dropna())

pairs["img_name_1"] = img_folder + get_image_name(pairs["ProductImageURL"]) + ".jpg"
pairs["img_name_2"] = img_folder + get_image_name(pairs["lookProductImage1"]) + ".jpg"

pairs = pairs[(pairs["img_name_1"].isin(imgs) & pairs["img_name_2"].isin(imgs))].reset_index(drop=True)

#run
main(train_batch_size = 256,
    test_batch_size = 256,
    epochs = 20, 
    optimizer_params = {"lr": 1e-5},     
    experiment_name = "base", 
    run_name = "_02", 
    pairs = pairs, 
    model_class = CLIPComplete)