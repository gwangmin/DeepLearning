'''
AutoEncoders
'''

from keras.models import Model
from keras.layers import Lambda, Input, Dense
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import numpy as np


class SimpleAE:
    '''
    Simple AutoEncoder
    Architecture: Input(input_dim) - Dense - Latent vector(latent_dim) - Dense - Output(input_dim)
    member:
        encoder: Encoder
        decoder: Decoder
        ae: Entire AutoEncoder
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', optimizer='adam', loss='mse'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, optimizer, loss)

    def build_model(self, input_dim, latent_dim, last_activation, optimizer, loss):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build ae
        self.ae = Model(inputs=self.input_x, outputs=self.decoder(self.encoder(self.input_x)))
        self.ae.compile(optimizer=optimizer, loss=loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,))
        encode = Dense(latent_dim, activation='relu')(self.input_x)
        self.encoder = Model(inputs=self.input_x, outputs=encode)

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,))
        decode = Dense(input_dim, activation=last_activation)(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode)


class SimpleDAE:
    '''
    Simple Denoising AutoEncoder
    Architecture: Input(input_dim) - Dense - Latent vector(latent_dim) - Dense - Output(input_dim)
    member:
        encoder: Encoder
        decoder: Decoder
        ae: Entire AutoEncoder
    method:
        add_random_noise(inputs[, mean, std, norm_clip])
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', optimizer='adam', loss='mse'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, optimizer, loss)

    def add_random_noise(self, inputs, mean=0, std=1, norm_clip=False):
        '''
        Add random noise to input
        (from normal dist.)
        '''
        noise = np.random.normal(loc=mean, scale=std, size=inputs.shape)
        result = inputs + noise
        if norm_clip:
            result = np.clip(result, 0., 1.)
        return result

    def build_model(self, input_dim, latent_dim, last_activation, optimizer, loss):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build ae
        self.ae = Model(inputs=self.input_x, outputs=self.decoder(self.encoder(self.input_x)))
        self.ae.compile(optimizer=optimizer, loss=loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,))
        encode = Dense(latent_dim, activation='relu')(self.input_x)
        self.encoder = Model(inputs=self.input_x, outputs=encode)

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,))
        decode = Dense(input_dim, activation=last_activation)(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode)


class SimpleVAE:
    '''
    Simple Variational AutoEncoder
    Architecture: Input - Dense - z_mean
                        - Dense - z_log_var
                        - Dense - z(sampled) - Dense - Output
    member:
        encoder: Encoder
        decoder: Decoder
        vae: Entire VAE
    args
        loss: 'mse' or 'bce'
    '''

    def __init__(self, input_dim, latent_dim, last_activation='sigmoid', loss='', optimizer='adam'):
        '''
        Initializer
        '''
        self.build_model(input_dim, latent_dim, last_activation, loss, optimizer)

    def build_model(self, input_dim, latent_dim, last_activation, loss, optimizer):
        '''
        Build entire autoencoder
        '''
        # Build encoder
        self.build_encoder(input_dim, latent_dim)
        # Build decoder
        self.build_decoder(input_dim, latent_dim, last_activation)
        # Build entire network
        outputs = self.decoder(self.encoder(self.input_x)[2])
        self.vae = Model(self.input_x, outputs, name='VAE')
        # VAE loss = mse_loss or xent_loss + kl_loss
        self.build_loss(input_dim, loss)
        # Compile
        self.vae.compile(optimizer=optimizer)

    def build_loss(self, input_dim, loss):
        '''
        Build VAE loss (= mse_loss or xent_loss + kl_loss)
        '''
        if loss == 'mse':
            reconstruction_loss = mse(self.input_x, self.vae(self.input_x))
        else:
            reconstruction_loss = binary_crossentropy(self.input_x, self.vae(self.input_x))
        #xent_loss = K.sum(K.binary_crossentropy(self.input_x, self.vae(self.input_x)), axis=-1)
        reconstruction_loss *= input_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

    def build_encoder(self, input_dim, latent_dim):
        '''
        Build encoder
        '''
        self.input_x = Input(shape=(input_dim,), name='input_x')
        self.z_mean = Dense(latent_dim, name='z_mean')(self.input_x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(self.input_x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(SimpleVAE.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.input_x, [self.z_mean, self.z_log_var, z], name='Encoder')

    def build_decoder(self, input_dim, latent_dim, last_activation):
        '''
        Build decoder
        '''
        input_latent = Input(shape=(latent_dim,), name='input_latent')
        decode = Dense(input_dim, activation=last_activation, name='decode')(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode, name='Decoder')

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        # K is the keras backend
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

