import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time
import h5py

import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Multiply, multiply, Softmax, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate
from tensorflow.keras.layers import Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot, Attention
from tensorflow.keras.models import Model

from custom.graphlayers import GCNLayer
from custom.transformer_layers import TokenAndPositionEmbedding, TransformerBlock
import tensorflow.keras as keras
import tensorflow.keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
import tensorflow.keras.backend as K
from model import create_model
from myutils import prep, drop, batch_gen, seq2sent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from keras.utils.vis_utils import plot_model
import tokenizer

#tf.compat.v1.disable_eager_execution()

class HistoryCallback(Callback):
    
    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig
        
        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)
    
    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = outdir+'/histories/'+self.modeltype+'_hist_'+str(self.timestart)+'.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)
                
                fn = outdir+'/histories/'+self.modeltype+'_conf_'+str(self.timestart)+'.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

class comb_gen(tensorflow.keras.utils.Sequence):
    def __init__(self, gen1, gen2):
        self.g1 = gen1
        self.g2 = gen2

    def __getitem__(self, idx):
        o1 = self.g1[idx]
        o2 = self.g2[idx]
        return ((o1[0], o2[0]), o1[1])

    def __len__(self):
        return int(np.ceil(np.array(self.g1.seqdata['dt%s' % (self.g1.tt)]).shape[0])/self.g1.batch_size)

if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile1', type=str, default=None)
    parser.add_argument('modelfile2', type=str, default=None)
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--with-calls', dest='withcalls', action='store_true', default=False)
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)
    parser.add_argument('--data', dest='dataprep', type=str, default='./data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir') 
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    parser.add_argument('--only-print-summary', dest='onlyprintsummary', action='store_true', default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=1337)
    parser.add_argument('--bagging', dest='bagging', action='store_true', default=False)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    withgraph = args.withgraph
    withcalls = args.withcalls
    vmemlimit = args.vmemlimit
    onlyprintsummary = args.onlyprintsummary
    seed = args.seed
    bagging = args.bagging
    modelfile1 = args.modelfile1
    modelfile2 = args.modelfile2
    #datfile = args.datfile

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    K.set_floatx(args.dtype)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if(vmemlimit > 0):
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
            except RuntimeError as e:
                print(e)

    prep('loading sequences... ')
    sqlfile = '{}/rawdats.sqlite'.format(dataprep)
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    seqdata = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    #seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
    drop()

    if withgraph:
        prep('loading graph data... ')
        graphdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
        for k, v in extradata.items():
            graphdata[k] = v
        extradata = graphdata
        drop()

    if withcalls:
        prep('loading call data... ')
        callnodes = pickle.load(open('%s/callsnodes.pkl' % (dataprep), 'rb'))
        calledges = pickle.load(open('%s/callsedges.pkl' % (dataprep), 'rb'))
        callnodesdata = pickle.load(open('%s/callsnodedata.pkl' % (dataprep), 'rb'))
        extradata['callnodes'] = callnodes
        extradata['calledges'] = calledges
        extradata['callnodedata'] = callnodesdata
        drop()

    prep('loading tokenizers... ')
    #tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    #comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    #sdatstok = pickle.load(open('%s/sdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    #smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = extradata['comstok']
    tdatstok = extradata['tdatstok']
    sdatstok = tdatstok
    smlstok = extradata['smlstok']
    if withgraph:
        graphtok = extradata['graphtok']
    drop()

    

    steps = int(np.array(seqdata.get('/ctrain').shape[0])/batch_size)#+1
    if bagging:
        steps = int(steps/2)
    #steps = 1
    valsteps = int(np.array(seqdata.get('/cval').shape[0])/batch_size)#+1
    #valsteps = 1
    
    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smlstok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*100))
    print('------------------------------------------')

    prep('loading config... ')
    tmp1 = modelfile1.split('/')
    tmp2 = modelfile2.split('/')
    outdir1 = '/'.join(tmp1[:-1])
    outdir2 = '/'.join(tmp2[:-1])
    modeltype1 = tmp1[-1]
    modeltype2 = tmp2[-1]
    (modeltype1, mid1, timestart1) = modeltype1.split('_')
    (modeltype2, mid2, timestart2) = modeltype2.split("_")
    (timestart1, ext1) = timestart1.split('.')
    (timestart2, ext2) = timestart2.split('.')
   # modeltype = modeltype.split('/')[-1]
   # modeltype2 = modeltype2.split('/')[-1]
    config1 = pickle.load(open(outdir1+'/'+modeltype1+'_conf_'+timestart1+'.pkl', 'rb'))
    config2 = pickle.load(open(outdir2+'/'+modeltype2+'_conf_'+timestart2+'.pkl', 'rb'))


    config1['bagging'] = False
    config2['bagging'] = False
    #comlen = config['comlen']
    #fid2loc = config['fidloc']['c'+testval] # fid2loc[fid] = loc
    #loc2fid = config['locfid']['c'+testval] # loc2fid[loc] = fid
    #allfids = list(fid2loc.keys())
    #allfidlocs = list(loc2fid.keys())

    drop()

    prep('loading model... ')
    model1 = keras.models.load_model(modelfile1, custom_objects={"tf":tf, "keras":keras, "GCNLayer":GCNLayer, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock':TransformerBlock})
    #print(model.summary())
    model2 = keras.models.load_model(modelfile2, custom_objects={"tf":tf, "keras":keras, "GCNLayer":GCNLayer, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock':TransformerBlock})
    #print(model2.summary())
    drop()

    for l in model1.layers:
        l._name = l.name+'_mdl1'
        l.trainable = False

    for l in model2.layers:
        l._name = l.name+'_mdl2'
        l.trainable = False
        print(l, l.trainable)

################################################
    # out = Attention()([model1.get_layer('time_distributed_mdl1').output, model2.get_layer('time_distributed_mdl2').output])
    # out = Flatten()(out)
    # out = Dense(100, activation='tanh')(out)

#################################################

    out = concatenate([model1.output, model2.output])
    out = Dense(100, activation='tanh')(out)
    out = Dense(config1['comvocabsize'], activation='softmax')(out)
    mdl = Model(inputs=[model1.input, model2.input], outputs=out)
    mdl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(mdl.summary())
    plot_model(mdl, to_file='model1.png', show_shapes=True, show_layer_names=True)
    #exit()
    gen = batch_gen(seqdata, extradata, 'train', config1, config2)

    #gen2 = batch_gen(seqdata, extradata, 'train', config2)
    #cg = comb_gen(gen1, gen2)
    #checkpoint = ModelCheckpoint(outdir+'/'+modeltype+'_E{epoch:02d}_TA{acc:.2f}_VA{val_acc:.2f}_VB{val_bleu:}.h5', monitor='val_loss')
    checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype1+'_'+modeltype2+'_E{epoch:02d}_'+str(timestart)+'.h5')
    savehist = HistoryCallback()
    savehist.setCatchExit(outdir, modeltype1+'_ens', timestart, config1)
    savehist.setCatchExit(outdir, modeltype2+'_ens', timestart, config2)
    
    valgen = batch_gen(seqdata, extradata, 'val', config1, config2)
    #valgen2 = batch_gen(seqdata, extradata, 'val', config2)
    #vg = comb_gen(valgen1, valgen2)
    # If you want it to calculate BLEU Score after each epoch use callback_valgen and test_cb
    #####
    #callback_valgen = batch_gen_train_bleu(seqdata, comvocabsize, 'val', modeltype, batch_size=batch_size)
    #test_cb = mycallback(callback_valgen, steps)
    #####
    callbacks = [ checkpoint, savehist ]

    try:
        history = mdl.fit(x=gen, steps_per_epoch=steps, epochs=epochs, validation_data=valgen, validation_steps=valsteps, verbose=1, max_queue_size=200, workers=20, callbacks=callbacks)#, validation_data=valgen, validation_steps=valsteps
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)
