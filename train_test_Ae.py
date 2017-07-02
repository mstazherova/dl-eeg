from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

cae = ConvolutionalAutoEncoder()
cae.train_from_dataset('datasets/CogReplay/dl-eeg/pgram_norm.hdf5', epochs= 11)



