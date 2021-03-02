import os#オペレーティングシステムとやり取りするための機能を備えているosモジュール、osモジュールインポートしたけど何しているの？
import sys#sysはPythonのインタプリタや実行環境に関する情報を扱うためのライブラリ,これも何に使っているの？
import random#ランダムな処理が必要な場合に活躍する標準モジュール
import time#時間を扱うことができるモジュール
#import tqdm#処理の進捗をプログレスバーで表示することができる
import numpy as np#https://qiita.com/jyori112/items/a15658d1dd17c421e1e2 #数値計算をより高速に、効率的に行うことができるようになります
import matplotlib.pyplot as plt#線グラフや棒グラフ、3Dグラフなどを描く
import pandas as pd#データ解析を行うための機能を持ったライブラリで、数表や時系列データを操作するためのデータ構造を作ったり演算を行う
import tensorflow as tf#多次元のデータ構造を、流れるように処理することができる深層学習（ディープラーニング）を行えるライブラリ
#https://udemy.benesse.co.jp/ai/tensorflow.html
#import GPy#入力データに対する非線形な回帰曲線を得たい
#https://kimbio.info/【python】gpyでガウス過程回帰と予測をやってみる
from keras import backend as K#Kerasとは、TensorFlow(※1)やTheano(※2)上で動くニューラルネットワークライブラリ(※3)の1つです。
#Kerasを使用すると、ディープラーニングのベースとなっている数学的理論の部分をゼロから開発せずとも、比較的短いソースコードで実装することができます。
#https://udemy.benesse.co.jp/ai/keras.html　#https://udemy.benesse.co.jp/ai/keras.html 
#以下kerasモジュールの各関数の役割が不明
from keras import activations, initializers
from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Layer, Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import Adam
#自分が行いたい分析（分類／回帰／クラスタリングなど）について、適切なモデルを選択する際の手助けとなる
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load, dump

# import My modules 

def plot_process(loss, val_loss):
    epochs = len(loss)#一つの訓練データを何回繰り返して学習させるか
    plt.yscale("log")#y-axisを対数表示に
    plt.plot(range(epochs), loss, marker='.', label='loss')#rangeの役割は？
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')#label凡例表示に必須
    plt.grid()#グリッドの設定https://www.pynote.info/entry/matplotlib-xticks-yticks-grid
    plt.xlabel('epoch')#x軸ラベル
    plt.ylabel('loss')#y軸ラベル
    plt.show()

def target_function(x):#この関数の意図は？
    return np.sin(x) + x#Numpyでsin関数を出力するときはnp.sin(x)とする。この場合はsin(x)+xという関数を用いる

if __name__=='__main__':#https://note.nkmk.me/python-if-name-main/
    # main process!
    # data
        # train
    x_train = np.random.randint(0, 10, 100).reshape(-1, 1)#開始0終了10個数100,reshapeの引数は？
    y_train = target_function(x_train).reshape(-1, 1)
        # test
    x_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test = target_function(x_test).reshape(-1, 1)
    
    # scaling :StandardScaler
    x_st = StandardScaler()
    y_st = StandardScaler()
        # train_dataにはfit_transform
    x_train_stsc = x_st.fit_transform(x_train)
    y_train_stsc = y_st.fit_transform(y_train)
        # test_dataにはtransform
    x_test_stsc = x_st.transform(x_test)
    y_test_stsc = y_st.transform(y_test)

    # neural networks
    nodes = 10
    model = Sequential()
        # input_layer
    model.add(Dense(nodes, input_dim=x_train.shape[1], activation='relu'))
        # hidden layer
    model.add(Dense(nodes, activation='relu'))
        # output layer
    model.add(Dense(y_train.shape[1]))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train_stsc, y_train_stsc, epochs=150, batch_size=2, verbose=2, validation_data=(x_test_stsc, y_test_stsc))
    model.save('model_1.h5')

    # plot learnig_process
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_process(loss, val_loss)

    # prediction by nn
    pred = model.predict(x_test_stsc)
    # predictionは標準化された値なので逆変換して元に戻す
    pred_inverse = y_st.inverse_transform(pred)
    # plot prediction
    plt.scatter(x_test, pred_inverse, label='prediction', s=1)
    plt.plot(x_test, y_test, label='target_function', c='red')
    plt.legend()
    plt.show()
