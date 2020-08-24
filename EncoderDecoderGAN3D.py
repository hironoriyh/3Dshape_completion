from __future__ import print_function, division

import os
from mpl_toolkits.mplot3d import Axes3D  # you should keep the import
import matplotlib.pyplot as plt

import numpy as np
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D, Deconv3D
from keras.models import Sequential, Model
from keras.models import load_model
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import hamming_loss
from utils import mkdirs

size = 64 #32

IMAGE_DIR = './%s_cube/images'%str(size)
MODEL_DIR = './%s_cube/saved_model'%str(size)
TESTDIR = os.path.join('%s_cube'%str(size), 'test')
           
mkdirs(IMAGE_DIR)
mkdirs(MODEL_DIR)
mkdirs(TESTDIR)

class EncoderDecoderGAN():

    def __init__(self):
        self.vol_rows = size
        self.vol_cols = size
        self.vol_height = size
        self.channels = 1
        self.num_classes = 2
        self.vol_shape = (self.vol_rows, self.vol_cols, self.vol_height, self.channels)

        optimizer = Adam(0.0002, 0.5)

        try:
            self.discriminator = load_model(os.path.join(MODEL_DIR, 'discriminator.h5'))
            self.generator = load_model(os.path.join(MODEL_DIR, 'generator.h5'))

            print("Loaded checkpoints")
        except:
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
            print("No checkpoints found")

            # discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # generator
        # The generator takes noise as input and generates the missing part
        masked_vol = Input(shape=self.vol_shape)
        gen_missing = self.generator(masked_vol)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated voxels as input and determines
        # if it is generated or if it is a real voxels
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_vol, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # Encoder
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.vol_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(128, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(512, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        model.add(UpSampling3D())
        model.add(Deconv3D(256, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Deconv3D(128, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling3D())
        model.add(Deconv3D(64, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling3D())
        model.add(Deconv3D(32, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling3D())

        model.add(Deconv3D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation('tanh'))
        model.summary()

        masked_vol = Input(shape=self.vol_shape)
        gen_missing = model(masked_vol)

        return Model(masked_vol, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv3D(32, kernel_size=3, strides=2, input_shape=self.vol_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        vol = Input(shape=self.vol_shape)

        validity = model(vol)

        return Model(vol, validity)

    def train(self, epochs, batch_size=16, sample_interval=50):

        # X_train = self.generateWall()
        
        X_train = np.load("data/train/all_vols.npy")
        X_masked_vols = np.load("data/train/all_masked_vols.npy")
        X_missing_parts = np.load("data/train/all_missing_parts.npy")
        print(X_train.shape)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        g_losses = []
        d_losses = []

        for epoch in range(epochs):

            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            vols = X_train[idx]

            # masked_vols, missing_parts, _ = self.mask_randomly(vols)
            masked_vols = X_masked_vols[idx]
            missing_parts = X_missing_parts[idx]
            # print(masked_vols.shape)
            # print(missing_parts.shape)

            # Generate a batch
            gen_vol = self.generator.predict(masked_vols)
            d_loss_real = self.discriminator.train_on_batch(vols, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_vol, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            g_loss = self.combined.train_on_batch(masked_vols, [vols, valid])

            g_losses.append(g_loss)
            d_losses.append(d_loss)

            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

            # save generated samples
            if epoch % sample_interval == 0:
                # idx = np.random.randint(0, X_train.shape[0], 2)
                # vols = X_train[idx]
                # self.sample_images(epoch, vols)
                self.save_model()

            # if epoch % 10 == 10-1:
                fig = plt.figure()
                fig = plt.figure(figsize=plt.figaspect(0.3))
                ax1 = fig.add_subplot(131, title='masked volume', projection='3d')
                ax2 = fig.add_subplot(132, title='valid missing', projection='3d') 
                ax3 = fig.add_subplot(133, title='gen combined', projection='3d') 

                ax1.voxels(masked_vols[0].reshape((size,size,size)).reshape((size,size,size)), facecolors='blue', edgecolor='k')
                ax2.voxels(missing_parts[0].reshape((size,size,size)), facecolors='green', edgecolor='k')
                
                gen_vol = np.where(gen_vol > 0.5, 1, 0)
                # print("gen_vol shape", gen_vol.shape)
                ax3.voxels(gen_vol[0].reshape((size,size,size)), facecolors='red', edgecolor='k')
                fig.savefig(os.path.join(IMAGE_DIR, "%d_keras.png" % epoch))
                np.save("%s_cube/losses_keras"%str(size), np.array(g_losses), np.array(d_losses))

                # plt.show()

    def test(self, path_, visualize = False):

        vols = np.load(os.path.join(path_, "all_vols.npy"))
        masked_vols = np.load(os.path.join(path_, "all_masked_vols.npy"))
        missing_parts = np.load(os.path.join(path_, "all_missing_parts.npy"))

        # vols = vols.reshape((1, size,size,size,1))
        # masked_vols = masked_vols.reshape((1,size,size,size,1))
        # missing_parts = missing_parts.reshape((1,size,size,size,1))

        gen_missing = self.generator.predict(masked_vols)
        gen_missing = np.where(gen_missing > 0.5, 1, 0)   
        combined_vols = masked_vols + gen_missing
        combined_vols = np.where(combined_vols > 0.9, 1, 0)   
        # print(np.where(combined_vols > 1))

        if(visualize):
            for i, vol in enumerate(vols):

                ### compute hamming loss
                one_gen_missing = gen_missing[i]
                one_gen_missing = one_gen_missing[:, :, :, 0].astype(np.bool)

                true_missing_part = missing_parts[i]
                true_missing_part = true_missing_part[:, :, :, 0].astype(np.bool)
                ham_loss = hamming_loss(true_missing_part.ravel(), one_gen_missing.ravel())
                print("hamming loss", ham_loss)

                fig = plt.figure()
                fig = plt.figure(figsize=plt.figaspect(0.25))
                fig.suptitle("Hamming Loss: %f" % ham_loss)
                ax1 = fig.add_subplot(141, title='masked volume', projection='3d')
                ax2 = fig.add_subplot(142, title='valid missing', projection='3d') 
                ax3 = fig.add_subplot(143, title='gen missing', projection='3d') 
                ax4 = fig.add_subplot(144, title='combined', projection='3d') 

                ax1.voxels(masked_vols[i].reshape((size,size,size)).reshape((size,size,size)), facecolors='blue', edgecolor='k')
                ax2.voxels(missing_parts[i].reshape((size,size,size)), facecolors='green', edgecolor='k')
                # print(gen_missing.shape)
                ax3.voxels(gen_missing[i].reshape((size,size,size)), facecolors='red', edgecolor='k')
                ax4.voxels(combined_vols[i].reshape((size,size,size)), facecolors='pink', edgecolor='k')
                fig.savefig(os.path.join(TESTDIR, "%d.png" % i))
        return combined_vols


    def save_model(self):
        def save(model, model_name):
            model_path = os.path.join(MODEL_DIR, "%s.h5" % model_name)
            model.save(model_path)

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    context_encoder = EncoderDecoderGAN()
    context_encoder.train(epochs=3000, batch_size=16, sample_interval=100)
