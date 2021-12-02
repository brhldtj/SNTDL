import numpy as np
import tensorflow as tf
from sklearn import metrics
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.models as Model

##### parameter settings  #####
sample = 6000
features = 9
classes = ['Microseismic events', 'Noise']

#### Set path  ####
data_path = './Datasets/'
x_train_savepath = data_path + 'X_train.npy'
y_train_savepath = data_path + 'Y_train.npy'
x_verify_savepath = data_path + 'X_verify.npy'
y_verify_savepath = data_path + 'Y_verify.npy'
x_test_savepath = data_path + 'X_test.npy'
y_test_savepath = data_path + 'Y_test.npy'

x_train_save = np.load(x_train_savepath)
y_train = np.load(y_train_savepath)
x_train = np.reshape(x_train_save, (len(x_train_save), sample, features))

x_verify_save = np.load(x_verify_savepath)
y_verify = np.load(y_verify_savepath)
x_verify = np.reshape(x_verify_save, (len(x_verify_save), sample, features))

x_test_save = np.load(x_test_savepath)
y_test = np.load(y_test_savepath)
x_test = np.reshape(x_test_save, (len(x_test_save), sample, features))

y_train = tf.one_hot(y_train, depth=len(classes))
y_verify = tf.one_hot(y_verify, depth=len(classes))
y_test = tf.one_hot(y_test, depth=len(classes))


data_format = 'channels_first'


def residual_block(Xm, filters, pool_size, strides_pool):
    # 1*1 Conv Linear
    ch = filters / 2

    Xm_shortcut = Conv2D(filters, kernel_size=(1, 1), strides=(2, 1), padding='same', activation="relu",
                         kernel_initializer='he_normal', data_format=data_format)(Xm)
    # Residual Unit 1
    Xm_1 = Conv2D(ch, kernel_size=(1, 1), strides=(1, 1), padding='same', activation="relu",
                  kernel_initializer='he_normal', data_format=data_format)(Xm)
    Xm_1 = Conv2D(ch, kernel_size=(3, 1), strides=(2, 1), padding='same',
                  kernel_initializer='he_normal', data_format=data_format)(Xm_1)
    Xm_1 = Conv2D(ch, kernel_size=(1, 1), padding='same', activation="relu",
                  kernel_initializer='glorot_normal', data_format=data_format)(Xm_1)

    Xm_2 = Conv2D(ch, kernel_size=(1, 1), strides=(1, 1), padding='same', activation="relu",
                  kernel_initializer='he_normal', data_format=data_format)(Xm)
    Xm_2 = MaxPooling2D(pool_size=(3, 1), strides=(2, 1), padding='same', data_format=data_format)(Xm_2)

    X_all = tf.concat([Xm_1, Xm_2], axis=1)
    Xm = tf.keras.layers.add([X_all, Xm_shortcut])
    Xm = Activation("relu")(Xm)

    Xm_shortcut = Xm

    # Residual Unit 2
    Xm_1 = Conv2D(ch, kernel_size=(1, 1), strides=(1, 1), padding='same', activation="relu",
                  kernel_initializer='he_normal', data_format=data_format)(Xm)
    Xm_1 = Conv2D(ch, kernel_size=(3, 1), strides=(1, 1), padding='same',
                  kernel_initializer='he_normal', data_format=data_format)(Xm_1)
    Xm_1 = Conv2D(ch, kernel_size=(1, 1), padding='same', activation="relu",
                  kernel_initializer='glorot_normal', data_format=data_format)(Xm_1)

    Xm_2 = Conv2D(ch, kernel_size=(1, 1), strides=(1, 1), padding='same', activation="relu",
                  kernel_initializer='he_normal', data_format=data_format)(Xm)
    Xm_2 = MaxPooling2D(pool_size=(3, 1), strides=(1, 1), padding='same', data_format=data_format)(Xm_2)

    X_all = tf.concat([Xm_1, Xm_2], axis=1)
    Xm = tf.keras.layers.add([X_all, Xm_shortcut])
    #     Xm = BatchNormalization(momentum=0.99, epsilon=0.001)(Xm)
    Xm = Activation("relu")(Xm)

    Xm = MaxPooling2D(pool_size=pool_size, strides=strides_pool, padding='valid', data_format=data_format)(Xm)
    return Xm


in_shp = (sample, features)

# input layer
Xm_input = Input(in_shp)
Xm = Reshape([1, sample, features], input_shape=in_shp)(Xm_input)

Xm = residual_block(Xm, filters=16, pool_size=(3, 1), strides_pool=(2, 1))
Xm = residual_block(Xm, filters=32, pool_size=(3, 1), strides_pool=(2, 1))
Xm = residual_block(Xm, filters=64, pool_size=(3, 1), strides_pool=(2, 1))
Xm = residual_block(Xm, filters=128, pool_size=(3, 1), strides_pool=(2, 1))
# Full Con 1
Xm = Flatten(data_format=data_format)(Xm)
Xm = Dense(256, activation='relu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.5)(Xm)
# Full Con 2
Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense3")(Xm)
# SoftMax
Xm = Activation('softmax')(Xm)
# Create Model
model = Model.Model(inputs=Xm_input, outputs=Xm)
adam = keras.optimizers.Adam(lr=0.001, decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)  # 参数优化器
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
model.summary()

print(tf.test.gpu_device_name())
filepath = "./Models/Model.ckpt"
history = model.fit(x_train, y_train,
                    batch_size=20,
                    epochs=50,
                    verbose=1,
                    validation_data=(x_verify, y_verify),
                    validation_freq=1,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto')
                    ])

test_Y_hat = model.predict(x_test, batch_size=32)
y_pred = np.zeros([x_test.shape[0],1])
y_true = np.zeros([x_test.shape[0],1])
for i in range(0, x_test.shape[0]):
    y_pred[i, :] = int(np.argmax(test_Y_hat[i,:]))
    y_true[i, :] = int(np.argmax(y_test[i,:]))


#  Precision
print('precision =', metrics.precision_score(y_true, y_pred, average='weighted'))
#  Recall
print('recall =', metrics.recall_score(y_true, y_pred, average='weighted'))
#  Accuracy
print('accuracy =', metrics.accuracy_score(y_true, y_pred, normalize=True))
# F1_score
print('f1_score =', metrics.f1_score(y_true, y_pred, average='weighted'))