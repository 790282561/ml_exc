import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image


class GAN():
    def __init__(self):
        self.latent_dim = 100
        self.height = 28
        self.width = 28
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)
        self.epochs = 4000
        self.save_dir = 'pics'

        # 配置生成器
        self.generator = self.build_generator()

        # 配置鉴别器
        self.discriminator = self.build_discriminator()
        discriminator_optimizer = keras.optimizers.RMSprop(
            lr=0.0008,
            clipvalue=1.0,
            decay=1e-8)
        self.discriminator.compile(
            optimizer=discriminator_optimizer,
            loss='binary_crossentropy')
        self.discriminator.trainable = False

        # 配置组装网络
        self.gan = self.build_gan()
        gan_optimizer = keras.optimizers.RMSprop(
            lr=0.0004,
            clipvalue=1.0,
            decay=1e-8)
        self.gan.compile(
            optimizer=gan_optimizer,
            loss='binary_crossentropy')

    def build_generator(self):
        # 构建生成器
        generator_input = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(128 * 14 * 14)(generator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((14, 14, 128))(x)

        x = layers.Conv2D(128, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # Conv2DTranspose是Conv2D的反向操作
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(self.channels, 7, activation='tanh', padding='same')(x)
        generator_model = models.Model(generator_input, x)
        # genenrator.summary()
        return generator_model

    def build_discriminator(self):
        # 构建判别器
        discriminator_input = layers.Input(shape=self.shape)
        x = layers.Conv2D(128, 3)(discriminator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        x = layers.Dropout(0.4)(x)

        x = layers.Dense(1, activation='sigmoid')(x)

        discriminator_model = models.Model(discriminator_input, x)
        # discriminator.summary()
        return discriminator_model

    def build_gan(self):
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gen_model = keras.models.Model(gan_input, gan_output)
        return gen_model

    def train(self):
        # 载入并归一化数据
        (train_data, train_labels), (_, _) = mnist.load_data()
        train_data = train_data.reshape((-1, 28, 28, 1))
        train_data = train_data.astype('float32') / 255.

        # 制造样本标签
        valid = np.ones((128, 1))
        fake = np.zeros((128, 1))

        # 训练生成
        for epoch in range(self.epochs):
            # 拆分样本
            '''
            从数据集随机挑选128个数据，作为一个批次训练。定义batch_size
            得到的结果是(128, 28, 28, 1)，其实是将样本分解为batch_size
            '''
            idx = np.random.randint(0, train_data.shape[0], 128)
            imgs = train_data[idx]

            # 制造噪音
            '''
            噪音维度(batch_size, 100)，并以此生成图像
            我不理解为什么必须要用正态分布的随机向量生成图像
            '''
            noise = np.random.normal(0, 1, (128, self.latent_dim))
            gen_img = self.generator.predict(noise)

            # 训练判别器，判别器希望真实图片打上标签1，假图打上0
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_img, fake)

            # 混合真假结果
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            g_loss = self.gan.train_on_batch(noise, valid)

            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

            # 每25个epoch保存一个生成图片
            if epoch % 25 == 0:
                self.gan.save_weights('gan.h5')  # 为什么要保存权重

                img = image.array_to_img(gen_img[0] * 255., scale=False)
                img.save(os.path.join(self.save_dir, 'generated_num' + str(epoch) + '.png'))

                img = image.array_to_img(imgs[0] * 255., scale=False)
                img.save(os.path.join(self.save_dir, 'real_num' + str(epoch) + '.png'))


if __name__ == '__main__':
    gan = GAN()
    gan.train()