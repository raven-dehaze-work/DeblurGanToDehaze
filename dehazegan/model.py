"""
实现去雾GAN类
"""
from functools import partial

from keras.layers import Input, Activation, Add, UpSampling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from dehazegan.custom_layers import RandomWeightedAverage
from dehazegan.layer_utils import ReflectionPadding2D,res_block
from dehazegan.losses import wasserstein_loss, perceptual_loss, gradient_penalty_loss


class DehazeNet:
    def __init__(self, batch_size) -> None:
        super().__init__()

        # 定义图片shape
        self.image_shape = (256, 256, 3)

        # 设置filter参数
        self.ngf = 64
        self.ndf = 64

        # gp 权重
        GRADIENT_PENALTY_WEIGHT = 10

        # build generator
        self.generator = self._build_generator(n_blocks_gen=9)

        # build pre_discriminator
        # 这里的discriminator不是后期用于训练的discriminator
        discriminator = self._build_pre_discriminator()

        # 上述的g和d是分离的，现在需要重新建立一个model，将两个model绑定
        # 同时编译generator
        # 首先fix住discriminator
        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False

        generator_input = Input(shape=self.image_shape,name='generator_input')
        generator_layers = self.generator(generator_input)
        discriminator_layers_for_generator = discriminator(generator_layers)
        # generator_model 包含了生成器部分和判别器部分。训练时可直接使用
        self.generator_model = Model(inputs=[generator_input],
                                     outputs=[generator_layers,discriminator_layers_for_generator])
        # 编译
        self.generator_model.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                                     loss=[perceptual_loss, wasserstein_loss],
                                     loss_weights=[100, 1])

        # 现在开放discriminator，关闭generator
        for layer in discriminator.layers:
            layer.trainable = True
        for layer in self.generator.layers:
            layer.trainable = False
        discriminator.trainable = True
        self.generator.trainable = False

        # 构建真的discriminator
        real_samples = Input(shape=self.image_shape,name='real_sample_input')
        generator_input_for_discriminator = Input(shape=self.image_shape,name='generator_input_for_discriminator')
        generated_samples_for_discriminator = self.generator(generator_input_for_discriminator)
        # discriminator输出
        discriminator_output_from_real_samples = discriminator(real_samples)
        discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)

        # We also need to generate weighted-averages of real and generated samples,
        # to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage(batch_size)([real_samples,
                                                              generated_samples_for_discriminator])
        # We then run these samples through the discriminator as well. Note that we never
        # really use the discriminator output for these samples - we're only running them to
        # get the gradient norm for the gradient penalty loss.
        averaged_samples_out = discriminator(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get
        # gradients. However, Keras loss functions can only have two arguments, y_true and
        # y_pred. We get around this by making a partial() of the function with the averaged
        # samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        # 建立 discriminator 的 model
        self.discriminator_model = Model(inputs=[real_samples,
                                                 generator_input_for_discriminator],
                                         outputs=[discriminator_output_from_real_samples,
                                                  discriminator_output_from_generator,
                                                  averaged_samples_out])

        self.discriminator_model.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                                         loss=[wasserstein_loss,
                                               wasserstein_loss,
                                               partial_gp_loss])

    def _build_generator(self, n_blocks_gen):
        """
        建造generator
        :return:
        """
        """Build generator architecture."""
        # Current version : ResNet block
        inputs = Input(shape=self.image_shape)

        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filters=self.ngf, kernel_size=(7, 7), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            x = Conv2D(filters=self.ngf * mult * 2, kernel_size=(3, 3), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        mult = 2 ** n_downsampling
        # nb_filter = self.ngf * mult
        for i in range(n_blocks_gen):
            x = res_block(x, self.ngf * mult, use_dropout=True)
            # x, nb_filter = dense_block(x, nb_layers=4, nb_filter=nb_filter, growth_rate=ngf * mult,
            #                    bottleneck=False, dropout_rate=0.5)
            # TODO: 是否有必要加transition_block？
            # x = transition_block(x, nb_filter)
        # 最后一层的denseblock 不一样，单独处理
        # x, nb_filter = dense_block(x, nb_layers=4, nb_filter=nb_filter, growth_rate=ngf * mult,
        #                    bottleneck=True, dropout_rate=0.5)

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
            x = UpSampling2D()(x)
            x = Conv2D(filters=int(self.ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=3, kernel_size=(7, 7), padding='valid')(x)
        x = Activation('tanh')(x)

        outputs = Add()([x, inputs])
        # outputs = Lambda(lambda z: K.clip(z, -1, 1))(x)
        outputs = Lambda(lambda z: z / 2)(outputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator')
        return model

    def _build_pre_discriminator(self):
        """Build discriminator architecture."""
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.image_shape)

        x = Conv2D(filters=self.ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            x = Conv2D(filters=self.ndf * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        x = Conv2D(filters=self.ndf * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid', name='d-output')(x)

        model = Model(inputs=inputs, outputs=x, name='Discriminator')
        return model


if __name__ == '__main__':
    DehazeNet(2)
