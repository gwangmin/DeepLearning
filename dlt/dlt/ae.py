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
        Build network
        '''
        input_x = Input(shape=(input_dim,))
        encode = Dense(latent_dim, activation='relu')(input_x)
        self.encoder = Model(inputs=input_x, outputs=encode)

        input_latent = Input(shape=(latent_dim,))
        decode = Dense(input_dim, activation=last_activation)(input_latent)
        self.decoder = Model(inputs=input_latent, outputs=decode)

        last_layer = self.decoder.layers[-1](encode)
        self.ae = Model(inputs=input_x, outputs=last_layer)
        self.ae.compile(optimizer=optimizer, loss=loss)

