import os
import DatasetBuilder
import matplotlib.pyplot as plt
import BinaryClassifier
from keras import callbacks
import tensorflow as tf

images_path = "C:/Users/allan/Downloads/CelebA/img_align_celeba/img_align_celeba"
attributes_file_path = "C:/Users/allan/Downloads/CelebA/list_attr_celeba.csv"
validation_folder = "C:/Users/allan/Downloads/GANFacesDateset"
positive_directory_path = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Positive"
negative_directory_path = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Negative"
possible_image_names = os.listdir(validation_folder)
batch_size = 128
target_size = (128, 128)
model_input_shape = (128,128,3)
crop_images = False
epochs = 5

def plot_image(tensor, classification):
    if classification == 0:
        classification = "Not Blonde"
    else:
        classification = "Blonde"
    plt.imshow(tensor)
    plt.title(classification)
    plt.show()
    pass





dataset_builder = DatasetBuilder.dataset_builder(
    images_path=images_path,
    attributes_file_path=attributes_file_path,
    batch_size=batch_size,
    target_size=target_size,
    crop_images=crop_images,
    attribute_name='Eyeglasses'
)

complete_dataset = dataset_builder.get_datasets()



classifier_builder = BinaryClassifier.BinaryClassifier(target_shape=model_input_shape)
classifier_builder.build_model()
classifier_builder.summarize_model()
classifier_builder.compile_model()

classifier = classifier_builder.get_model()

def validation_callback(epoch, logs=None):
    tf.keras.models.save_model(classifier, f"EyeGlassesModels/Epoch{epoch+1}.h5")
    # for i in range(10):
    #     #Get an image name
    #     image_index = random.randrange(0, len(possible_image_names))
    #     #Construct the image path
    #     image_path = os.path.join(validation_folder, possible_image_names[image_index])
    #     #Read an image from file
    #     image = tf.keras.utils.load_img(image_path)
    #     #Conver the file into an array
    #     input_arr = tf.keras.utils.img_to_array(image)
    #     #Scale it
    #     input_arr = input_arr/255.
    #     #Resize to target size
    #     input_arr = tf.image.resize(input_arr, target_size)
    #     #Batch it
    #     input_tensor = tf.expand_dims(input_arr, axis=0)
    #     #Get a prediction
    #     classification = classifier(input_tensor)[0]
    #     #Convert it into a format to be able to make a prediction
    #     classification = np.array(classification)
    #     if classification > 0.5:
    #         prediction = 1
    #     else:
    #         prediction = 0
    #     #Plot the image
    #     plot_image(input_arr, prediction)
    #     pass
    pass

my_callback = callbacks.LambdaCallback(on_epoch_end=validation_callback)


classifier.fit(x = complete_dataset, epochs = epochs, callbacks=[my_callback])




