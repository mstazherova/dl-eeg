__author__ = 'Steffen'

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

model = ConvolutionalAutoEncoder.model()
model.load_weights('cae_weights.h5')

weights =