import BuildGlassesDataset
import matplotlib.pyplot as plt
import tensorflow as tf
import BinaryClassifier
from keras import callbacks

positive_directory_path = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Positive"
negative_directory_path = "C:/Users/allan/Downloads/CelebA/Eyeglasses_dataset/Negative"
target_size = (128, 128)
model_input_shape = (128,128,3)
crop_images = False
batch_size = 32
epochs = 20
train_val_split = 0.95

BuildGlassesDataset = BuildGlassesDataset.BuildGlassesDataset(
    target_size=target_size,
    crop_images=crop_images,
    positive_directory_path=positive_directory_path,
    negative_directory_path=negative_directory_path,
    batch_size=batch_size,
    train_val_split=train_val_split
)

train_dataset, val_dataset = BuildGlassesDataset.get_datasets()




classifier_builder = BinaryClassifier.BinaryClassifier(target_shape=model_input_shape)
classifier_builder.build_model()
classifier_builder.summarize_model()
classifier_builder.compile_model()

classifier = classifier_builder.get_model()

def validation_callback(epoch, logs=None):
    tf.keras.models.save_model(classifier, f"EyeGlassesModels2/Epoch{epoch+1}.h5")
    pass

my_callback = callbacks.LambdaCallback(on_epoch_end=validation_callback)


classifier.fit(x = train_dataset, epochs = epochs,validation_data = val_dataset, callbacks=[my_callback])