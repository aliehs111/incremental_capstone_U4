# Project Notes: Incremental Capstone Unit 4

## 2025-06-05
 - used os.getcwd() to show project root
 - - Opened `processing.ipynb` and realized the notebook was running from `notebooks/`, so I added: os.chdir("..") to go up a directory

  
 - Loaded the images with image_dataset_from_directory("images", …, validation_split=0.2), splitting 80% train / 20% validation:

 - - Got 791 total images (633 train, 158 validation).

 - - Confirmed there are 20 training batches and 5 validation batches.

 - Created a Rescaling(1.0/255) layer and applied it to both train_ds and val_ds so all pixel values are scaled to [0, 1].

 - Ran a quick visualization cell in processing.ipynb to display a 3×3 grid of normalized images (no more clipping warnings).


 - preprocessing steps (loading, splitting, normalizing, augmenting, caching/prefetching) are complete—train_ds and val_ds are ready to feed into a CNN.

Next step: switch to train_model.ipynb to build, compile, and train the convolutional model.
