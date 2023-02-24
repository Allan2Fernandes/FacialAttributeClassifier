import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
class BuildGlassesDataset:
    def __init__(self, target_size, crop_images, positive_directory_path, negative_directory_path, batch_size, train_val_split):

        positive_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=positive_directory_path,
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

        negative_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=negative_directory_path,
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
        #Create the dataset and scale it
        positive_dataset = positive_dataset.map(self.map_function)#.batch(batch_size= batch_size, drop_remainder = True)
        negative_dataset = negative_dataset.map(
            self.map_function)  # .batch(batch_size=batch_size, drop_remainder = True)
        #Create labels
        positive_labels = tf.ones(shape=[len(positive_dataset)])
        negative_labels = tf.zeros(shape=[len(negative_dataset)])
        positive_labels = tf.data.Dataset.from_tensor_slices(positive_labels)
        negative_labels = tf.data.Dataset.from_tensor_slices(negative_labels)
        #zip the datasets
        positive_dataset = tf.data.Dataset.zip((positive_dataset, positive_labels))
        negative_dataset = tf.data.Dataset.zip((negative_dataset, negative_labels))

        complete_dataset = positive_dataset.concatenate(negative_dataset)

        # Split the dataset into train and val
        # get the length of the dataset
        dataset_length = len(complete_dataset)
        train_length = int(dataset_length * train_val_split)
        complete_dataset = complete_dataset.shuffle(buffer_size=len(complete_dataset))
        self.train_dataset = complete_dataset.take(train_length)
        self.validation_dataset = complete_dataset.skip(train_length)

        self.train_dataset = self.train_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
        self.validation_dataset = self.validation_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)


        pass

    def map_function(self, datapoint):
        datapoint = (datapoint) / 255.
        return datapoint

    def get_datasets(self):
        return self.train_dataset, self.validation_dataset