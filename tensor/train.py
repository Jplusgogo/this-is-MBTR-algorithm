import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from MBTR import create_mobilevit

batch_size = 32
num_classes = 7
epochs = 2
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "--filepath--", "--filename--")  # flower data set path
train_dir = os.path.join(image_path, "train")
validation_dir = os.path.join(image_path, "val")
assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)



def pre_function(img):

    return img

# data generator with data augmentation
train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=pre_function)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(image_size, image_size),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(image_size, image_size),
                                                              class_mode='categorical')
# img, _ = next(train_data_gen)
total_val = val_data_gen.n
print("using {} images for training, {} images for validation.".format(total_train,
                                                                       total_val))

# create model except fc layer
model = create_mobilevit(num_classes=num_classes)
model.summary()

writer = tf.summary.create_file_writer('/path/to/log', tf.compat.v1.get_default_graph())

# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)

@tf.function
def val_step(images, labels):
    output = model(images, training=False)
    loss = loss_object(labels, output)

    val_loss(loss)
    val_accuracy(labels, output)

best_val_acc = 0.
for epoch in range(epochs):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    val_loss.reset_states()  # clear history info
    val_accuracy.reset_states()  # clear history info

    # train
    train_bar = tqdm(range(total_train // batch_size))
    for step in train_bar:
        images, labels = next(train_data_gen)
        train_step(images, labels)

        # print train process
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                             epochs,
                                                                             train_loss.result(),
                                                                             train_accuracy.result())

    # validate
    val_bar = tqdm(range(total_val // batch_size))
    for step in val_bar:
        val_images, val_labels = next(val_data_gen)
        val_step(val_images, val_labels)

        # print val process
        val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                           epochs,
                                                                           val_loss.result(),
                                                                           val_accuracy.result())

    # only save best weights
    if val_accuracy.result() > best_val_acc:
        best_val_acc = val_accuracy.result()
        model.save_weights("./save_weights/mbvit-1cktf.ckpt", save_format="tf")
        model.save_weights("./save_weights/mbvit-1ckh5.ckpt", save_format="h5")
        model.save_weights("./save_weights/mbvit-1h5h5.h5", save_format="h5")

writer.close()

