import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

class GAN():
    def __init__(self, input_shape, depth):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.input_shape = input_shape
        self.depth = depth
        self.dim = 14
        self.generator = self.make_gen()
        
        #these two lines are necesarry or the generator is unable to call
        #predict later in the code. I do not know why, possibly calling
        #predict compiles and saves the model?
        noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
        self.generator.predict(noise)
        
        self.discriminator, self.f_discriminator = self.make_descs()
        self.adversarial = self.compile_adv(self.generator, self.f_discriminator)
        self.real_images = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.real_images = self.real_images.reshape(-1, self.img_rows,self.img_cols, 1).astype(np.float32)

    def make_gen(self):
        model = Sequential()
        model.add(Dense((self.dim**2)*self.depth, input_dim=100))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((self.dim, self.dim, self.depth)))
        model.add(Dropout(0.5))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int((self.depth*4)/2), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(int((self.depth*4)/4), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(int((self.depth*4)/8), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(1, 5, padding='same'))
        model.add(Activation('sigmoid'))
        return model
        
    def make_descs(self):
        optimizer = RMSprop(lr=0.00008, clipvalue=1.0, decay=6e-8)
        x = Input(shape=self.input_shape)
        layer1 = Conv2D(self.depth, 5, strides=2, padding='same', activation='relu')
        layer1.trainable = True
        layer2 = Dropout(0.5)
        layer2.trainable = True
        layer3 = Conv2D(self.depth*2, 5, strides=2, padding='same', activation='relu')
        layer3.trainable = True
        layer4 = Dropout(0.5)
        layer4.trainable = True
        layer5 = Conv2D(self.depth*4, 5, strides=2, padding='same', activation='relu')
        layer5.trainable = True
        layer6 = Dropout(0.5)
        layer6.trainable = True
        layer7 = Conv2D(self.depth*8, 5, strides=2, padding='same', activation='relu')
        layer7.trainable = True
        layer8 = Dropout(0.5)
        layer8.trainable = True
        layer9 = Flatten()
        layer9.trainable = True
        layer10 = Dense(1, activation='sigmoid')
        layer10.trainable = True
        y = layer10(layer9(layer8(layer7(layer6(layer5(layer4(layer3(layer2(layer1(x))))))))))
        model = Model(x,y)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',\
                      metrics=['accuracy'])
        layer1.trainable = False
        layer2.trainable = False
        layer3.trainable = False
        layer4.trainable = False
        layer5.trainable = False
        layer6.trainable = False
        layer7.trainable = False
        layer8.trainable = False
        layer9.trainable = False
        layer10.trainable = False
        f_model = Model(x,y)
        f_model.compile(optimizer=optimizer, loss='binary_crossentropy',\
                      metrics=['accuracy'])
        return model, f_model

    def compile_adv(self, gen, desc):
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        model = Sequential()
        model.add(gen)
        model.add(desc)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, \
                      metrics=['accuracy'])
        return model

    def genDescData(self):
        for i in range(self.batch_size):
            real_images = self.real_images[np.random.randint(0,self.real_images.shape[0], size=self.batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            fake_images = self.generator.predict(noise)
            x = np.concatenate((real_images, fake_images))
            y = np.concatenate((np.ones([self.batch_size, 1]), np.zeros([self.batch_size, 1])))
            yield x, y
        return

    def genAdvData(self):
        for i in range(self.batch_size):
            y = np.ones([self.batch_size, 1])
            x = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            yield x, y
        return
            
    def train(self, epochs=10, batch_size=32):
        self.batch_size = batch_size
        for i in range(epochs):
            desc_loss = self.discriminator.fit_generator(self.genDescData(), steps_per_epoch=1,\
                                                         epochs=1, verbose=1)
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            adv_loss = self.adversarial.fit_generator(self.genAdvData(), steps_per_epoch=1,\
                                                      epochs=1, verbose=1)
        return
            

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "mnist_%d.png" % step
        fake_images = self.generator.predict(noise)

        i = np.random.randint(0, self.real_images.shape[0], samples)
        images = self.real_images[i, :, :, :]

        plt.figure(figsize=(6,6))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            if i < 8:
                image = images[i, :, :, :]
            else:
                image = fake_images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()    

if __name__=='__main__':
    GAN = GAN((28,28,1), 64)
    GAN.train(5000, 265)
    GAN.plot_images(fake=True)
    #GAN.plot_images(fake=False, save2file=False)
