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
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, AUC

from src.chestX_net import ChestXrayNet
from src.dataset import TfdataPipeline
from src.losses import WeightedBinaryCrossEntropy, SigmoidFocalLoss


__supported_models__ = ['resnet50', 'densenet121', 'mobilenetv3', 'efficientnet']
__supported_losses__ = ['w_binary_crossentropy', 'sigmoid_focal_loss']

def train(
    dataset_base_dir:str,
    train_csv_filename:str = 'train_labels.csv',
    validation_csv_filename:str = 'valid_labels.csv',
    checkpoint_dir:str='save_model/',
    pretrained_model_checkpoint:str=None,
    IMG_H:int = 224,
    IMG_W:int = 224,
    IMG_C:int = 3,
    batch_size:int = 16,
    lr:float = 1e-4,
    epochs:int = 100,
    base_model_name:str = 'resnet50',
    loss_name:str = 'w_binary_crossentropy',
    do_augmentation:bool = True,
    trainable_layers:int = 10,
):
    assert base_model_name in __supported_models__, f'{base_model_name} is not supported, supported models are {__supported_models__}'
    assert loss_name in __supported_losses__, f'{loss_name} is not supported, supported losses are {__supported_losses__}'
    assert os.path.exists(dataset_base_dir), f'{dataset_base_dir} does not exist'
    assert os.path.exists(os.path.join(dataset_base_dir, train_csv_filename)), f'{train_csv_filename} does not exist in {dataset_base_dir}.\
        Check {dataset_base_dir} for the correct train_csv_filename'
    assert os.path.exists(os.path.join(dataset_base_dir, validation_csv_filename)), f'{validation_csv_filename} does not exist in {dataset_base_dir}.\
        Check {dataset_base_dir} for the correct validation_csv_filename'

    # Create Checkpoint Directory if not exists
    os.mkdir(checkpoint_dir) if not os.path.exists(checkpoint_dir) else None

    # Create Dataset
    tf_dataset = TfdataPipeline(
        BASE_DATASET_DIR = dataset_base_dir,
        IMG_H=IMG_H,
        IMG_W=IMG_W,
        IMG_C=IMG_C,
        batch_size=batch_size
    )

    train_ds = tf_dataset.load_dataset(train_csv_filename, do_augmemt=do_augmentation)
    valid_ds = tf_dataset.load_dataset(validation_csv_filename, do_augmemt=False)

    # Intantiate Optimizer
    optimizer = Adam(learning_rate=lr)

    # Instantiate Losses
    if loss_name == 'w_binary_crossentropy':
        loss = WeightedBinaryCrossEntropy()
    elif loss_name == 'sigmoid_focal_loss':
        loss = SigmoidFocalLoss()
    else:
        print(f'{loss_name} is not supported, supported losses are {__supported_losses__}')


    # Create Model
    if pretrained_model_checkpoint is not None:
        assert os.path.exists(pretrained_model_checkpoint), f'{pretrained_model_checkpoint} does not exist'
        model = models.load_model(pretrained_model_checkpoint, custom_objects={'SigmoidFocalLoss': SigmoidFocalLoss})
    else:
        model = ChestXrayNet(inshape=(IMG_H, IMG_W, IMG_C), base_model_name=base_model_name, num_classes=14, trainable_layers=trainable_layers)

    # Compile Model
    model.compile(optimizer=optimizer, loss=loss, metrics=[BinaryAccuracy(), AUC()])

    # Create Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{base_model_name}-{loss_name}')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')

    # Train Model
    model.summary()
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    # Save Model
    model.save(os.path.join(checkpoint_dir, f'final_model-{base_model_name}-{loss_name}'), save_format='tf')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base_dir', type=str, required=True, help='Base Directory of the Dataset')
    parser.add_argument('--pretrained_model_checkpoint', type=str, default=None, help='Pretrained Model Checkpoint')
    parser.add_argument('--train_csv_filename', type=str, default='train_labels.csv', help='Train CSV Filename')
    parser.add_argument('--validation_csv_filename', type=str, default='valid_labels.csv', help='Validation CSV Filename')
    parser.add_argument('--checkpoint_dir', type=str, default='save_model/', help='Checkpoint Directory')
    parser.add_argument('--IMG_H', type=int, default=224, help='Image Height')
    parser.add_argument('--IMG_W', type=int, default=224, help='Image Width')
    parser.add_argument('--IMG_C', type=int, default=3, help='Image Channels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--base_model_name', type=str, default='resnet50', help='Base Model Name')
    parser.add_argument('--loss_name', type=str, default='w_binary_crossentropy', help='Loss Name')
    parser.add_argument('--do_augmentation', type=int, default=1, help='Do Augmentation', choices=[0, 1])
    args = parser.parse_args()

    train(
        dataset_base_dir=args.dataset_base_dir,
        pretrained_model_checkpoint=args.pretrained_model_checkpoint,
        train_csv_filename=args.train_csv_filename,
        validation_csv_filename=args.validation_csv_filename,
        checkpoint_dir=args.checkpoint_dir,
        IMG_H=args.IMG_H,
        IMG_W=args.IMG_W,
        IMG_C=args.IMG_C,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        base_model_name=args.base_model_name,
        loss_name=args.loss_name,
        do_augmentation=bool(args.do_augmentation)
    )