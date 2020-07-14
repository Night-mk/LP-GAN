# LP-GAN
lightweight Privacy-Preserving GAN Framework

## dataset
dataset is saved in:
```python
/data/MNIST
/data/Cifar-10
/data/CelebA
```

## model
model is saved in:
```python
/model_save
```

## secure computation protocols
secure computation protocols can be found in:
```python
/layers/secure_protocols.py
/layers/secure_protocols_2.py
/layers/secure_protocols_3.py
```

## secure protocols for different layers in Neural Network
```python
/layers/Conv_sec.py # Convolution
/layers/Deconv_sec.py # DeConvolution
/layers/FC_sec.py # Fully Connected
/layers/BN_sec.py # Batch Normalization
/layers/Logsoftmax_sec.py # softmax+log
/layers/Activator_sec.py # square, ReLU, LeakyReLU, Sigmoid, Tanh
/layers/Loss_sec.py # Cross Entropy
```

## secure inference and training on different networks
secure inference and training code can be found in:
```python
/layers/Secureml_5.py # 3-FC,square
/layers/CryptoNet_5.py # 1-Conv,2-FC,square
/layers/Minionn_5.py # 1-Conv,2-FC,ReLU
```

## LP-GAN test
the secure image synthesis and testing work can be found in:
```python
/layers/DCGAN_mnist_sec.py
```
