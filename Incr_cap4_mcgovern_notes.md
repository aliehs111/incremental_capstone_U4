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

## 2025-06-07
- **Refreshed environment**: Restarted kernel and consolidated imports in Cell 1, including:
  - `os.chdir("..")` to set project root
  - `tensorflow`, `layers`, `models`, `callbacks`
  - `from data_utils import get_datasets`
  - `numpy` and `matplotlib.pyplot`
- **Loaded datasets** by calling `train_ds, val_ds = get_datasets()`, verified 20 training and 5 validation batches.
- **Built and summarized CNN** (three Conv→BatchNorm→Pool blocks, Flatten→Dense(128)→Dropout→Dense(1, sigmoid)).
- **Compiled model** with Adam (lr=1e-4), binary crossentropy, and set up:
  - `EarlyStopping(patience=5, restore_best_weights=True)`
  - `ModelCheckpoint('best_model.h5', save_best_only=True)`
- **Trained** for up to 20 epochs; saw validation accuracy peak at ~90.5% around epoch 13 and early stop around epoch 18.
- **Evaluated** best model:
  - Accuracy: **0.9051**, Loss: **0.7564**
  - Confusion matrix:  
    ```
    [[77  6]
     [ 9 66]]
    ```
  - Precision/Recall/F1 for “Bike” and “Car”: ~0.91 / ~0.90 / ~0.90
- **Plotted** training vs. validation accuracy & loss curves.
- **Displayed** sample predictions (3×3 grid) with true vs. predicted labels and confidence scores.

*Next:* Analyze overfitting/underfitting from the plots, tweak hyperparameters or architecture as needed, then draft the final report.  

## 2025-06-09
- tried several different combinations of hyperparameter tuning and rebuilding the layers of the model.  The results were either overfitting or just poor accuracy.
- In the end, taking out the batch normalization layers (all 3 of them) and that seemed to do the trick.  Final Results:
Final Train Acc:      0.9068
Final Validation Acc: 0.8228
Accuracy Gap:         0.0840
Final Train Loss:     0.2649
Final Validation Loss:0.3950
Loss Gap:             0.1300

Confusion Matrix:
[[  0   0]
 [ 45 113]]

Classification Report:
              precision    recall  f1-score   support

        Bike       0.00      0.00      0.00         0
         Car       1.00      0.72      0.83       158

    accuracy                           0.72       158
   macro avg       0.50      0.36      0.42       158
weighted avg       1.00      0.72      0.83       158