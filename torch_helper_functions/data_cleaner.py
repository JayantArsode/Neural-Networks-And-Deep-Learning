"""
Contains functionality for cleaning data related to data kept in folders and it's labels are in dataframe.

## Functions:
  `preprocess_image_paths`:Preprocesses image paths in a DataFrame by replacing parts of the paths and changing extensions.
  `valid_image_paths`:Preprocesses image paths in a DataFrame by removing all the invalid non-existing images present in the original DataFrame.
"""
import os
import pandas as pd
import numpy as np
def preprocess_image_paths(df:pd.DataFrame,
                           image_path_column: str,
                           orignal_path: str,
                           new_path: str,
                           old_extension: str,
                           new_extension: str = None,
                           new_image_path_column: str = None,):
  """
  #### Preprocesses image paths in a DataFrame by replacing parts of the paths and changing extensions.

  ## Parameters:
    `df (pd.DataFrame)`: The DataFrame containing image paths.
    `image_path_column (str)`: The name of the column in df containing the image paths.
    `new_image_path_column (str)`: The name of the column in df where to save the new image paths
                                           if none it will just replce orignal path.
    `original_path (str)`: The part of the image path to be replaced in dataframe diffrent from your current data location.
    `new_path (str)`: The replacement for the original_path in dataframe for new data location.
    `old_extension (str)`: The extension of the image paths to be replaced.
    `new_extension (str)`: The new extension to replace the old extension if none keeps same extension.
  ## Returns:
    None: The function modifies the input DataFrame in place and does not return any value.
  ## Example:
      preprocess_image_paths(df,"image_paths","D://kaggle//","D://root","jpg","jpeg","new_image_paths")
  """
  if new_image_path_column is None:
    new_image_path_column = image_path_column
  if new_extension is None:
    new_extension = old_extension
  df[new_image_path_column] = df[image_path_column].str.replace(orignal_path, new_path)
  df[new_image_path_column] = df[image_path_column].str.replace(old_extension, new_extension)

def validate_image_paths(df: pd.DataFrame,
                         image_path_column: str,
                         folder_path: str,
                         save_path: str="root"):
  """
  #### Preprocesses image paths in a DataFrame by removing all the invalid non-existing images present in the original DataFrame.

  ## Parameters:
    `df (pd.DataFrame)`: The DataFrame containing image paths.
    `image_path_column (str)`: The name of the column in df containing the image paths.
    `folder_path (str)`: The path to the folder containing the images.
    `save_path (str)=default="root"`: The path to save a numpy file contatining list of valid images if not path just filename.

  ## Returns:
    None: The function modifies the input DataFrame in place and does not return any value
            also save a numpy file containing valid images names.
  ## Example:
     validate_image_paths(df,"image_paths","D://train_image_data","D://valid_image//valid_image_paths")
  """
  valid_images = []
  for index, row in df.iterrows():
    image_path = os.path.join(folder_path, row[image_path_column])
    # Check if the file exists
    if os.path.isfile(image_path):
      valid_images.append(row[image_path_column])
  if os.path.isfile(save_path):
    print("Valid Image File Already Exists")
  else:
    valid_images = np.array(valid_images)
    np.save(save_path,valid_images)
  # Filter the DataFrame to include only rows with valid image paths
  df = df[df[image_path_column].isin(np.load(save_path))]
