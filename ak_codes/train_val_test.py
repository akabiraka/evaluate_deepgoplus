import sys
sys.path.append("../deepgoplus")

import numpy as np
import pandas as pd
from utils import Ontology
import pickle_utils
from aminoacids import to_onehot, MAXLEN
import tensorflow as tf
import logging
import math
import click as ck
from sklearn.metrics import roc_curve, auc

from keras.utils import Sequence
from keras.optimizers import adam_v2
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, RepeatVector, Layer
)

from keras.utils import Sequence
from keras import backend as K
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from config import Config
from ak_codes.test import run_test
from ak_codes.val import run_val

config = Config()
model_name = config.get_model_name()

species = config.species #"yeast"
GOname= config.GO #"BP"
data_generation_process = config.data_generation_process #"time_delay_no_knowledge"


class DFGenerator_(Sequence):
    def __init__(self, dataset, terms_dict, batch_size):
        self.start = 0
        self.batch_size = batch_size
        self.seq_db_dict = pickle_utils.load_pickle(f"ak_data/{species}.pkl")

        self.dataset = dataset  # [uniprot_id, set(terms)]
        self.size = len(self.dataset)

        self.terms_dict = terms_dict
        self.nb_classes = len(self.terms_dict)
        
    
    def __len__(self):                                                                                                                   
        return np.ceil(len(self.dataset) / float(self.batch_size)).astype(np.int32)  

    
    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size: min(self.size, (idx + 1) * self.batch_size)]
        data_onehot = np.zeros((len(batch), MAXLEN, 21), dtype=np.float32)
        labels = np.zeros((len(batch), self.nb_classes), dtype=np.int32)

        for i, (uniprot_id, terms_set) in enumerate(batch):
            seq = self.seq_db_dict[uniprot_id]["seq"]
            onehot = to_onehot(seq)
            data_onehot[i, :, :] = onehot

            for term in terms_set:
                labels[i, self.terms_dict[term]] = 1
            
        self.start += self.batch_size
        return (data_onehot, labels)

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch = self.dataset[self.start: min(self.size, self.start + self.batch_size)]
            data_onehot = np.zeros((len(batch), MAXLEN, 21), dtype=np.float32)
            labels = np.zeros((len(batch), self.nb_classes), dtype=np.int32)

            for i, (uniprot_id, terms_set) in enumerate(batch):
                seq = self.seq_db_dict[uniprot_id]["seq"]
                onehot = to_onehot(seq)
                data_onehot[i, :, :] = onehot

                for term in terms_set:
                    labels[i, self.terms_dict[term]] = 1
                
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def create_model(nb_classes, params):
    inp_hot = Input(shape=(MAXLEN, 21), dtype=np.float32)
    
    kernels = range(8, params['max_kernel'], 8)
    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='valid',
            name='conv_' + str(i),
            kernel_initializer=params['initializer'])(inp_hot)
        print(conv.get_shape())
        pool = MaxPooling1D(
            pool_size=MAXLEN - kernels[i] + 1, name='pool_' + str(i))(conv)
        flat = Flatten(name='flat_' + str(i))(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    for i in range(params['dense_depth']):
        net = Dense(nb_classes, activation='relu', name='dense_' + str(i))(net)
    net = Dense(nb_classes, activation='sigmoid', name='dense_out')(net)
    model = Model(inputs=inp_hot, outputs=net)
    model.summary()
    model.compile(
        optimizer=params['optimizer'],
        loss=params['loss'])
    print('Compilation finished')

    return model


def main(batch_size=32, epochs=12, device='gpu:0', params_index=-1, load=False):
        params = {
            'max_kernel': 129,
            'initializer': 'glorot_normal',
            'dense_depth': 0,
            'nb_filters': 512,
            'optimizer': adam_v2.Adam(learning_rate=3e-4),
            'loss': 'binary_crossentropy'
        }
        # SLURM JOB ARRAY INDEX
        pi = params_index
        if params_index != -1:
            kernels = [33, 65, 129, 257, 513]
            dense_depths = [0, 1, 2]
            nb_filters = [32, 64, 128, 256, 512]
            params['max_kernel'] = kernels[pi % 5]
            pi //= 5
            params['dense_depth'] = dense_depths[pi % 3]
            pi //= 3
            params['nb_filters'] = nb_filters[pi % 5]
            pi //= 5
            out_file = f'ak_outputs/{model_name}_predictions_{params_index}.pkl'
            logger_file = f'ak_outputs/{model_name}_training_{params_index}.csv'
            model_file = f'ak_outputs/{model_name}_model_{params_index}.h5'
        print('Params:', params)


        # go = Ontology("ak_data/go.obo", with_rels=True)

        terms_dict = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/studied_terms.pkl")
        nb_classes = len(terms_dict)

        train_dataset = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/train.pkl")
        train_steps = int(math.ceil(len(train_dataset) / batch_size))
        train_generator = DFGenerator_(train_dataset, terms_dict, batch_size)

        valid_dataset = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/val.pkl")
        valid_steps = int(math.ceil(len(valid_dataset) / batch_size))
        valid_generator = DFGenerator_(valid_dataset, terms_dict, batch_size)

        test_dataset = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/test.pkl")
        test_steps = int(math.ceil(len(test_dataset) / batch_size))
        test_generator = DFGenerator_(test_dataset, terms_dict, batch_size)
        
        print(f"Training data size: {train_generator.__len__()}")
        print(f"Validation data size: {valid_generator.__len__()}")
        print(f"Test data size: {test_generator.__len__()}")


        with tf.device('/' + device):
            if load:
                print('Loading pretrained model')
                model = load_model(model_file)
            else:
                print('Creating a new model')
                model = create_model(nb_classes, params)
                
                checkpointer = ModelCheckpoint(
                    filepath=model_file,
                    verbose=1, save_best_only=True)
                earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
                logger = CSVLogger(logger_file)

                print('Starting training the model')
                # model.summary()
                model.fit(
                    train_generator,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=valid_steps,
                    max_queue_size=batch_size,
                    workers=12,
                    callbacks=[logger, checkpointer, earlystopper])
                print('Loading best model')
                model = load_model(model_file)

        
            print('Evaluating model')

            valid_generator.reset()
            preds = model.predict(valid_generator, steps=valid_steps)
            run_val(preds)

            test_generator.reset()
            preds = model.predict(test_generator, steps=test_steps)
            run_test(preds)
            


if __name__ == '__main__':
    main(batch_size=32, epochs=12, device='gpu:0', params_index=1, load=True)
