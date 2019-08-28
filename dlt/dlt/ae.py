'''
AutoEncoders
'''

from keras.models import Model
from keras.layers import Input, Dense


class SimpleAE:
    '''
    Simple AutoEncoder
    Architecture: Input(input_dim) - Dense - Latent vector(latent_dim) - Dense - Output(input_dim)
    member:
        obj.encoder: Encoder
        obj.decoder: Decoder
        obj.ae: Entire AutoEncoder
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

