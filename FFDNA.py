import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix,f1_score, roc_auc_score
from keras.layers import Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.layers import Reshape, Dropout, Add, Subtract, Flatten, Embedding
#from keras.optimizers import Adam

from keras.models import load_model, Model
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")

from process_data import *


#import feather
#%matplotlib inline

class FFDNA(object):
    
    def __init__(self,data, config):
        self.time_decay_tr = data[0]
        self.time_decay_te = data[1]
        self.X_tr = data[2]
        self.X_te = data[3]
        self.X_tr_lr = data[4]
        self.X_te_lr = data[5]
        self.Y_train = data[6]
        self.Y_test = data[7]
        self.all_X = data[8]
        self.time_decay = data[9]
        self.newlines = data[10]
        self.y = data[11]
        self.categorical_vars = data[12]
        self.paths = data[13]
        self.config = config
        self.channels = config['channels']
        self.Tx = config['Tx']
        self.n_a = config['n_a']
        self.n_s = config['n_s']
        self.s0 = config['s0']
        self.s1 = config['s1']
        self.vocab_size = config['vocab_size']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        
    def save_weight(self,name,model):
        model.save_weights(name)
    
    def load_weight(self,name):
        self.model.load_weights(name)
      
    def one_step_attention(self, a, s_prev,t0):
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation = "tanh")
        densor2 = Dense(1, activation = "relu")
        activator = Activation(self.softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        dotor = Dot(axes = 1)
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a".
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = concatenator([s_prev,a])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" 
        energies = Subtract(name='data-time')([energies,t0])
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next layer
        context = dotor([alphas,a])
        return context 
    
    
    def build_embedding_network(self, no_of_unique_cat=83, output_shape=32):
        inputss = []
        embeddings = []
        for c in self.categorical_vars:
            inputs = Input(shape=(1,),name='input_sparse_'+c)
            #no_of_unique_cat  = data_lr[categorical_var].nunique()
            embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
            embedding_size = int(embedding_size)
            embedding = Embedding(no_of_unique_cat+1, embedding_size, input_length = 1)(inputs)
            embedding = Reshape(target_shape=(embedding_size,))(embedding)
            inputss.append(inputs)
            embeddings.append(embedding)
        input_numeric = Input(shape=(1,),name='input_constinuous')
        embedding_numeric = Dense(16)(input_numeric)
        inputss.append(input_numeric)
        embeddings.append(embedding_numeric)

        x = Concatenate()(embeddings)

        x = Dense(10, activation = 'relu')(x)
        x = Dropout(.15)(x)
        out_control = Dense(output_shape)(x)
        return inputss,out_control

    def softmax(self,x,axis=1):
        ndim = K.ndim(x)
        if ndim==2:
            return K.softmax(x)
        elif ndim >2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims =True)
            return e/s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')
    
    def model(self):
        '''
        模型初始化

        '''
        # Define the inputs of your model with a shape (Tx,)
        # Define s0, initial hidden state for the decoder LSTM of shape (n_s,)
        input_att = Input(shape=(self.Tx, self.vocab_size), name='input_path')
        s0 = Input(shape=(self.n_s,), name='s0')
        s = s0
        # input time decay data
        t0 = Input(shape=(self.Tx,1), name='input_timeDecay')
        t = t0
        # Step 1: Define pre-attention LSTM.
        a = LSTM(self.n_a,return_sequences=True)(input_att)
 
        # Step 2: import attention model
        context = self.one_step_attention(a,s,t)
        c = Flatten()(context)
        out_att = Dense(32, activation = "sigmoid", name='single_output')(c)

        # Step 3: import embedding data for customer-ralated variables
        input_con,out_control = self.build_embedding_network()
        added = Add()([out_att, out_control])
        out_all = Dense(1,activation='sigmoid')(added)
        # Step 4: Create model instance taking three inputs and returning the list of outputs.
        self.model = Model([input_att,s0,t0,input_con[0],
                      input_con[1],input_con[2],input_con[3]],out_all)
        print(self.model.summary())
        #return self.model
    
    def train_model(self,save_name, loss='binary_crossentropy',opt='adam',metrics=['accuracy']):
        self.model.compile(loss=loss,optimizer=opt,metrics=metrics)
        self.history = self.model.fit([self.X_tr,self.s0,self.time_decay_tr,self.X_tr_lr.iloc[:,0],self.X_tr_lr.iloc[:,1],
               self.X_tr_lr.iloc[:,2],self.X_tr_lr.iloc[:,3]
              ], self.Y_train, epochs=self.epochs, batch_size=self.batch_size,verbose=2)
    
        self.save_weight(save_name,self.model)
    
    # model performance
    def plot_roc_curve(self, fpr, tpr, label=None): 
        plt.plot(fpr, tpr, linewidth=2, label=label) 
        plt.plot([0, 1], [0, 1], 'k--') 
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    def metric(self, y_valid,prob,cl,label=None):
        fpr, tpr, threshold = roc_curve(y_valid, prob)
        auc = roc_auc_score(y_valid,prob)
        self.plot_roc_curve(fpr,tpr,label=label)
        acc = (y_valid==cl).mean()
        print('Accuracy: {:.3f}, AUC: {:.3f}'.format(acc,auc))
    

    def plot_loss(self):
        ylims = range(1,self.epochs+1,10)
        plt.plot(self.history.history['loss'],color='red',label='train loss')
        plt.xticks(ylims)
        plt.legend(loc=1)
        plt.title('train loss vs epochs')
    def plot_acc(self):
        ylims = range(1,self.epochs+1,10)
        plt.plot(self.history.history['acc'],label='acc',c='r')    
        plt.xticks(ylims)
        plt.legend(loc=4)
        plt.title('train acc vs epochs')
    def auc_score_train(self,threshold):
        prob = self.model.predict([self.X_tr,self.s0,self.time_decay_tr,self.X_tr_lr.iloc[:,0],self.X_tr_lr.iloc[:,1],
               self.X_tr_lr.iloc[:,2],self.X_tr_lr.iloc[:,3]])
        cl = [1 if p > threshold else 0 for p in prob]
        print(confusion_matrix(self.Y_train,cl))
        print(self.metric(self.Y_train,prob,cl,label='train dataset performance'))
    def auc_score_test(self,threshold):
        prob = self.model.predict([self.X_te,self.s1,self.time_decay_te,self.X_te_lr.iloc[:,0],self.X_te_lr.iloc[:,1],
               self.X_te_lr.iloc[:,2],self.X_te_lr.iloc[:,3]])
        cl = [1 if p > threshold else 0 for p in prob]
        print(confusion_matrix(self.Y_test,cl))
        print(self.metric(self.Y_test,prob,cl,label='test dataset performance'))  
        
    def test_model(self,threshold,train=False):
        if train:
            self.auc_score_train(threshold)
        else:
            self.auc_socre_test(threshold)
            
    # credits for different channels; as the input data for budget calculation formula
    def attributes(self):
        layer = self.model.layers[20]
        m_all,_,_ = self.all_X.shape
        self.s_all = np.zeros((m_all, self.n_s))
        f_f = K.function([self.model.input[0],self.model.input[1],self.model.input[2]], [layer.output])
        r=f_f([self.all_X[self.y==1],self.s_all[self.y==1],self.time_decay[self.y==1]])[0].reshape(self.all_X[self.y==1].shape[0],self.all_X[self.y==1].shape[1])
        
        att_f = {m:0 for m in range(1,6)}
        att_count_f = {m:0 for m in range(1,6)}
        chan_used = self.newlines[self.y==1]
        for m in range(chan_used.shape[0]):
            for n in range(chan_used.shape[1]):
                if chan_used[m,n]!=0:
                    att_f[chan_used[m,n]] += r[m,n]
                    att_count_f[chan_used[m,n]] += 1
        att_f[self.channels[0]] = att_f.pop(1)
        att_f[self.channels[1]] = att_f.pop(2)
        att_f[self.channels[2]] = att_f.pop(3)
        att_f[self.channels[3]] = att_f.pop(4)
        att_f[self.channels[4]] = att_f.pop(5)
        
        return att_f
    
    def critical_paths(self):
        prob = self.model.predict([self.X_tr,self.s0,self.time_decay_tr,self.X_tr_lr.iloc[:,0],self.X_tr_lr.iloc[:,1],
           self.X_tr_lr.iloc[:,2],self.X_tr_lr.iloc[:,3]])
        cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
        #print([prob[p] for p in cp_idx[0:100]])
        cp_p = [self.paths[p] for p in cp_idx[0:100]]
        
        cp_p_2 = set(map(tuple, cp_p))
        print(list(map(list,cp_p_2)))
    
if __name__ == '__main__':
    #got data
    # data = pd.read_csv('df_paths_noblank_purchase.csv')
    data = pd.read_excel('fake_data.xlsx')
    seq_length = 20
    data_all = process_data(data,seq_length = seq_length)
    '''
    time_decay_tr, 训练集时间
    time_decay_te,测试集时间
    X_tr, 用户路径模块 -训练集X
    X_te,用户路径模块 - 测试集X
    X_tr_lr,用户属性模块-训练集X
    X_te_lr,用户属性模块-测试集X
    Y_train,
    Y_test, 
    all_X, 所有x
    time_decay,所有时间
    newlines.shape,17*20,pad_sequence
    y ,所有y
    categorical_vars,用户属性模块 哪些字段
    paths , 所有路径
    '''
    
    
    # hyper parameters
    n_a = 32
    n_s = 64
    m = data_all[2].shape[0]
    m_t = data_all[3].shape[0]
    s0 = np.zeros((m, n_s)) # 15*64
    s1 = np.zeros((m_t, n_s)) # 2*64
    batch_size = 64
    Tx = seq_length
    learning_rate = 0.001
    
    # channels = ['Natural Search','Email','Paid Search','Media','Social']
    channels = list(set([_p  for p in data_all[-1]  for _p in p ]))
    
    n_channels = len(channels)
    vocab_size = n_channels +1
    epochs = 120
    config = {'channels':channels, 'Tx':Tx, 'n_a':n_a, 'n_s':n_s, 's0':s0,'s1':s1,'vocab_size':vocab_size,
          'epochs':epochs,'batch_size':batch_size,'learning_rate':learning_rate}
    
    # model
    ana_mta_model = FFDNA(data_all, config)
    # 模型初始化
    ana_mta_model.model()
    
    # 模型训练
    save_name = 'FFDNA_full.h5'
    ana_mta_model.train_model(save_name, loss='binary_crossentropy',opt='adam',metrics=['accuracy'])
    
    # 模型重载
    ana_mta_model = FFDNA(data_all, config)
    ana_mta_model.model()
    ana_mta_model.load_weight('FFDNA_full.h5')
    
    # 训练集预测
    prob = ana_mta_model.model.predict([ana_mta_model.X_tr,ana_mta_model.s0,ana_mta_model.time_decay_tr,\
                                        ana_mta_model.X_tr_lr.iloc[:,0],\
                                            ana_mta_model.X_tr_lr.iloc[:,1],
       ana_mta_model.X_tr_lr.iloc[:,2],ana_mta_model.X_tr_lr.iloc[:,3]])
    
    
    # pred需要包括:
    #     - ana_mta_model.X_tr# 用户路径特征
    #     - ana_mta_model.s0 # 15*64 训练集
    #     - ana_mta_model.time_decay_tr # 训练集时间
    #     - ana_mta_model.X_tr_lr.iloc[:,0] # 用户属性特征 - marketing_area
    #     - ana_mta_model.X_tr_lr.iloc[:,1] # 用户属性特征 - tier
    #     - ana_mta_model.X_tr_lr.iloc[:,2] # 用户属性特征 - customer_type
    #     - ana_mta_model.X_tr_lr.iloc[:,3] # 用户属性特征 - c_type_na
    
        # 训练集预测 - 找到预测概率比较高的路径
    cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
    #print([prob[p] for p in cp_idx[0:100]])
    cp_p = [ana_mta_model.paths[p] for p in cp_idx[0:100]]
    
    cp_p_2 = set(map(tuple, cp_p))
    print(list(map(list,cp_p_2)))

    # 获得每个节点权重
    layer = ana_mta_model.model.layers[20]
    m_all,_,_ = ana_mta_model.all_X.shape  # 训练集
    ana_mta_model.s_all = np.zeros((m_all, ana_mta_model.n_s))
    f_f = K.function([ana_mta_model.model.input[0],ana_mta_model.model.input[1],ana_mta_model.model.input[2]], [layer.output])
    r=f_f([ana_mta_model.all_X[ana_mta_model.y==1],ana_mta_model.s_all[ana_mta_model.y==1],ana_mta_model.time_decay[ana_mta_model.y==1]])[0].reshape(ana_mta_model.all_X[ana_mta_model.y==1].shape[0],ana_mta_model.all_X[ana_mta_model.y==1].shape[1])
    
    
    # att_f = {m:0 for m in range(1,6)}
    # att_count_f = {m:0 for m in range(1,6)}
    att_f = {m:0 for m in range(1,n_channels+1)}
    att_count_f = {m:0 for m in range(1,n_channels+1)}
    
    chan_used = ana_mta_model.newlines[ana_mta_model.y==1]
    for m in range(chan_used.shape[0]):
        for n in range(chan_used.shape[1]):
            if chan_used[m,n]!=0:
                att_f[chan_used[m,n]] += r[m,n]
                att_count_f[chan_used[m,n]] += 1
    for n in range(n_channels):
        att_f[channels[n]] = att_f.pop(n+1)
        
    # att_f[m.channels[0]] = att_f.pop(1)
    # att_f[m.channels[1]] = att_f.pop(2)
    # att_f[m.channels[2]] = att_f.pop(3)
    # att_f[m.channels[3]] = att_f.pop(4)
    # att_f[m.channels[4]] = att_f.pop(5)



    # #m.train_model('s.h5')
    # print('\n\n 1. Test dataset performance:\n')
    # m.auc_score_test(0.5)
    # print('\n\n 2. Train performance:\n')
    # m.auc_score_train(0.5)
    # att_f = m.attributes()
    # print('\n\n 3. Channel credits: \n',att_f)
    # print('\n\n 3. Top critical paths: \n')
    # m.critical_paths()
    
    
    
    
    
    
    
    
    
    
    
    
    