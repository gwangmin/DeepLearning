'''
AutoEncoders
'''

from keras.models import Model
from keras.layers import Input, Dense
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
        add_noise(inputs[, mean, std, norm_clip])
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
            result = np.clip(inputs, 0., 1.)
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

