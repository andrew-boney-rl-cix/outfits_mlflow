#libraries
import os
import dataiku

from model.utils import save_img

#data/models
images = dataiku.Folder("F2uM1rcH")
all_img_paths = images.list_paths_in_partition()

#preproccess
##images
for p in all_img_paths: save_img(images, p, "code_studio-versioned/pics/")