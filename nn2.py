"""
neural_networks(nn)
"""
import numpy as np
# for nn
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import Adam
# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# save models
from joblib import dump, load
from random import randint
import random
import pandas as pd
# plot
import matplotlib.pyplot as plt


def plot_process(loss,  val_loss):
    epochs = len(loss)
    plt.yscale("log")
    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    #plt.savefig("image/plot_process_NEW_DEISGN2222.jpg")

#  ----------------学習データ------------------------------------------------------------------------------------------------
# data_path
train_data_path = "data/csv/train_data_NEW_DESIGN.csv"
test_data_path = "data/csv/val_data_NEW_DESIGN.csv"
# load_data
train_data = np.loadtxt(train_data_path, delimiter=",")
test_data = np.loadtxt(test_data_path, delimiter=",")
print(train_data[0])

#  ----------------標準化------------------------------------------------------------------------------------------------
# x,yの別々のscalerを用意する
x_st = StandardScaler() 
y_st = StandardScaler()
# scailng x
X_train_stsc = x_st.fit_transform(train_data[:, 1:12])
X_test_stsc = x_st.transform(test_data[:, 1:12])
# scailng y
Y_train_stsc = y_st.fit_transform(train_data[:, [12, 13, 14, 15, 16]])
Y_test_stsc = y_st.transform(test_data[:,  [12, 13, 14, 15, 16]])
print(np.max(X_train_stsc), np.min(Y_train_stsc))
print(np.max(X_test_stsc), np.min(Y_test_stsc))

#  ----------------標準化保存------------------------------------------------------------------------------------------------
# 好きなところに保存
dump(x_st, "stsc/stsc_x_NEW_DESIGN.joblib")
dump(y_st, "stsc/stsc_y_NEW_DESIGN.joblib")

#  ---------------define model and train--------------------------------------------------------------------------------
# この書き方も慣れ
# 活性化関数は、sigmoidは回帰性能が高い。reluは、回帰性能そこそこで学習が早い。
# l2_normは、1e-6がデフォルト。1e-3,1e-4,1e-5,1e-6を調べる。値が小さいほど回帰性能が高いが汎化性能が低くなる。過学習を抑えたいなら大きな値を設定する
model = Sequential()
# node数
nodes = 200
# 入力層
model.add(Dense(nodes, input_dim=11, activation='relu', activity_regularizer=regularizers.l2(1e-6)))
# 隠れ層
model.add(Dense(nodes, activation='relu', activity_regularizer=regularizers.l2(1e-6)))
model.add(Dense(nodes, activation='relu', activity_regularizer=regularizers.l2(1e-6)))
# 出力層
model.add(Dense(5))

# 最適化方法。基本はAdam。
optimizer = Adam(lr=0.0002)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')
# epoch数は適当。100~1000くらい？それより、batch_sizeが重要。少ないほど、学習に時間がかかるが、少ないエポックで学習が終了する。
history = model.fit(X_train_stsc, Y_train_stsc, epochs=400, batch_size=512, verbose=2,
                    validation_data=(X_test_stsc, Y_test_stsc), callbacks=[early_stopping])
# modelの保存。
model.save('model/model_NEW_DESIGN.h5')

# 学習過程をplotするコード
loss = history.history['loss']
val_loss = history.history['val_loss']
plot_process(loss, val_loss)

