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

from tensorflow.keras.losses import Loss, binary_crossentropy

# A Weighted Cross Entropy Loss for multi-label classification.
class WeightedBinaryCrossEntropy(Loss):
    def __init__(self, pos_w: float, from_logits:bool=False, name: str = 'weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(name=name)
        self.pos_w = pos_w
        self.neg_w = 1.0 - pos_w
        self.epsilon = 1e-8
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred:tf.Tensor)-> tf.Tensor:
        assert y_pred.shape == y_true.shape, f'Shape mismatch: y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}'

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.math.sigmoid(y_pred)
        
        # Compute the weighted cross entropy.
        loss = -(self.pos_w * y_true * tf.math.log(y_pred+self.epsilon) + self.neg_w * (1.0 - y_true) * tf.math.log(1.0 - y_pred+self.epsilon))

        return loss
    
    def get_config(self):
        config = super(WeightedBinaryCrossEntropy, self).get_config()
        config.update({
            'pos_w': self.pos_w,
            'neg_w': self.neg_w,
            'epsilon': self.epsilon,
            'from_logits': self.from_logits
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SigmoidFocalLoss(Loss):
    def __init__(self, alpha:float = 0.25, gamma:float = 2.0, from_logits:bool = False, name:str='focal_loss'):
        super(SigmoidFocalLoss, self).__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.epsilon = 1e-8
    
    def call(self, y_true:tf.Tensor, y_pred:tf.Tensor)->tf.Tensor:
        assert y_pred.shape == y_true.shape, f'Shape mismatch: y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}'

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.math.sigmoid(y_pred)

        # Compute the focal loss.
        loss = -(
            self.alpha * y_true * tf.math.pow(1.0 - y_pred, self.gamma) * tf.math.log(y_pred+self.epsilon) +
            (1.0 - self.alpha) * (1.0 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1.0 - y_pred+self.epsilon)
        )
        
        return loss
    
    def get_config(self):
        config = super(SigmoidFocalLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    
    dummy_label = tf.constant([ [0, 1, 1, 0], 
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 1, 0]], dtype=tf.float32)

    dumm_pred1 = 0.9 * tf.ones_like(dummy_label)
    dumm_pred2 = 0.1 * tf.ones_like(dummy_label)

    pos_w =  tf.reduce_sum(dummy_label, axis=0) / dummy_label.shape[0]

    wbce = WeightedBinaryCrossEntropy(pos_w)
    loss1 = wbce(dummy_label, dumm_pred1)
    loss2 = wbce(dummy_label, dumm_pred2)
    tf.print(f'BCE Loss1: {loss1}\nBCE Loss2: {loss2}')

    focal = SigmoidFocalLoss()
    loss3 = focal(dummy_label, dumm_pred1)
    loss4 = focal(dummy_label, dumm_pred2)
    tf.print(f'Focal Loss1: {loss3}\nFocal Loss2: {loss4}')