# Assignment: Classification of fake symbols.
# We will import the `load_and_predict()` function below to assess your assignment.

def load_and_predict(directory, model_file):
    # The `directory` argument is a folder with CSV files. 
    # It doesn *not* have subfolders because this function is meant to be used for prediction.
    # Example:
    # ```
    # /path/to/some/dir
    #   |_ file1.csv
    #   |_ file2.csv
    #   |_ ...
    # ```
    # The `model_file` argument is a trained model file, in `.pth` format.
    #
    # This function must implement the following steps:
    # (1) Read the data from the provided directory.
    # (2) Prepare the data according to whatever preprocessing pipeline you used during model training.
    # (3) Load your model checkpoint.
    # (4) Query the model with the data in (1) in order to get the predicted class probabilities for each instance.
    # (5) Convert probabilities to labels (integer numbers): 0 for "real" and 1 for "fake".
    # (6) Return a dictionary where keys are absolute file paths and values are the predicted labels for each file. Example:
    # ```
    # {
    #   '/path/to/some/dir/file1.csv': 0, 
    #   '/path/to/some/dir/file2.csv': 1, 
    #   ... 
    # }
    #```
    return pred_dict

