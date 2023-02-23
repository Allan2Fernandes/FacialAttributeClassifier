import tensorflow as tf
import pandas as pd


class dataset_builder:
    def __init__(self, images_path, attributes_file_path, target_size, crop_images, batch_size, attribute_name=None):
        image_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=images_path,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode='rgb',
            batch_size=None,
            image_size=target_size,
            shuffle=False,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=crop_images)

        image_dataset = image_dataset.map(self.map_function)

        data_frame = pd.read_csv(attributes_file_path)
        #Get all available attributes to be able to then classify
        list_of_attributes = data_frame.columns
        # Convert to numpy which is needed to convert to tensor
        classifications = data_frame[attribute_name].to_numpy()
        # Conver to binary values for classification
        classifications[classifications == -1] = 0
        # Convert to tensor
        classifications = tf.constant(classifications, tf.int32)
        # Convert to a dataset
        classifications_dataset = tf.data.Dataset.from_tensor_slices(classifications)

        self.complete_dataset = tf.data.Dataset.zip((image_dataset, classifications_dataset))
        self.complete_dataset = self.complete_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)

        pass

    def map_function(self, datapoint):
        datapoint = (datapoint) / 255.
        return datapoint


    def get_datasets(self):
        return self.complete_dataset


