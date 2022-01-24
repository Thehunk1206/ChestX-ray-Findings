'''
MIT License

Copyright (c) 2022 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV3Large, EfficientNetB4
from tensorflow.keras import layers


class ChestXrayNet(Model):
    def __init__(
        self,
        inshape:tuple = (224,224,3),
        base_model_name:str='desnet121',
        base_model_weights:str='imagenet',
        base_model_trainable:bool=False,
        num_classes:int=14,
        trainable_layers:int = 10,
        *args, 
        **kwargs
    ):
        super(ChestXrayNet, self).__init__(*args, **kwargs)

        self.inshape = inshape
        self.base_model_name = base_model_name
        self.base_model_weights = base_model_weights
        self.base_model_trainable = base_model_trainable
        self.num_classes = num_classes
        self.trainable_layers = trainable_layers

        base_model_dict = {
            'resnet50': ResNet50,
            'densenet121': DenseNet121,
            'mobilenetv3': MobileNetV3Large,
            'efficientnet': EfficientNetB4
        }

        self.base_model = base_model_dict[base_model_name](
            weights=self.base_model_weights,
            include_top=False,
            input_shape=self.inshape
        )
        for i in range(len(self.base_model.layers) - self.trainable_layers):
            self.base_model.layers[i].trainable = self.base_model_trainable

        self.global_average_pooling = layers.GlobalAveragePooling2D(keepdims=True)
        self.flatten = layers.Flatten()
        self.dense_layer = layers.Dense(self.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs:tf.Tensor, training=None):
        if self.base_model_name in ['resnet50', 'densenet121']:
            inputs = layers.Rescaling(1./255)(inputs)
        x = self.base_model(inputs)
        x = self.global_average_pooling(x)
        x = self.flatten(x)
        x = self.dense_layer(x)
        x = tf.nn.sigmoid(x)
        return x
    
    def summary(self, *args, **kwargs):
        x = layers.Input(shape=self.inshape)
        model = Model(inputs=[x], outputs=self.call(x), name='ChestXrayNet')
        model.summary(*args, **kwargs)
    
    def get_config(self):
        return {
            'inshape': self.inshape,
            'base_model_name': self.base_model_name,
            'base_model_weights': self.base_model_weights,
            'base_model_trainable':self.base_model_trainable,
            'num_classes': self.num_classes,
            'trainable_layers': self.trainable_layers
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    input_shape = (224,224,3)
    base_model_name = 'efficientnet'
    dumm_input = tf.random.normal(shape=(8,224,224,3))

    model = ChestXrayNet(
        inshape=input_shape,
        base_model_name=base_model_name,
    )
    out = model(dumm_input)
    tf.print(out.shape)
    model.summary()
    print(model.base_model.layers[:10])
