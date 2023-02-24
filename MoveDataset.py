import pandas as pd
import shutil
import os


base_directory_path = "C:/Users/allan/Downloads/CelebA/img_align_celeba/img_align_celeba"
attributes_file_path = "C:/Users/allan/Downloads/CelebA/list_attr_celeba.csv"
positive_directory = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Positive"
negative_directory = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Negative"
def copy_over_image(source_path, destination_path):
    shutil.copy(source_path, destination_path)



# data_frame = pd.read_csv(attributes_file_path)
#
# data_frame = data_frame[["image_id", "Eyeglasses"]]
#
# filtered_dataframe = data_frame[data_frame["Eyeglasses"] == -1]

num_positives = len(os.listdir(positive_directory))
num_negatives = len(os.listdir(negative_directory))

print(num_positives)
print(num_negatives)


# for index, name in enumerate(filtered_dataframe["image_id"]):
#     source_path = os.path.join(base_directory_path, name)
#     destination_path = os.path.join(negative_directory, name)
#     copy_over_image(source_path, destination_path)
#     if index%500 == 0:
#         print(f"Completed copying over: {index}/{num_positives}")
#         pass
#     if index >= num_positives:
#         break



#
#
#
# for index, name in enumerate(filtered_dataframe["image_id"]):
#     source_path = os.path.join(base_directory_path, name)
#     destination_path = os.path.join(positive_directory, name)
#     copy_over_image(source_path, destination_path)
#     if index%500 == 0:
#         print(f"Completed copying over: {index}/{len(filtered_dataframe)}")
#     pass

