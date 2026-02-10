#该代码用于LCAN，TDIT，MT的实验
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import normalize
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
# 创建回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# 加载多个数据文件
def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1].values  # 假设最后一列是标签
    return X, y

# 加载数据
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values  # 假设最后一列是标签
    return X, y

# 自定义自注意力层
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                  shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                  shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.exp(e)
        a = e / K.sum(e, axis=1, keepdims=True)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

# 构建模型
def build_model(input_shape, num_labels):
    model = Sequential()
    # Input layer
    model.add(Conv1D(filters=8, kernel_size=80, strides=20, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    # 添加自注意力层
    model.add(SelfAttention())
    model.add(Flatten())  # 或者使用 GlobalMaxPooling1D()
    model.add(Dense(128, activation='relu'))  # 特征层
    model.add(Dense(num_labels, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param classes: 类别名称列表
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def calculate_macro_metrics(y_true, y_pred, num_classes):
    """
    计算宏平均精确率、召回率和F1分数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param num_classes: 类别数量
    :return: 宏平均精确率、召回率和F1分数
    """
    precision_macro = np.zeros(num_classes)
    recall_macro = np.zeros(num_classes)
    f1_macro = np.zeros(num_classes)

    for i in range(num_classes):
        TP = np.sum((y_pred == i) & (y_true == i))
        FP = np.sum((y_pred == i) & (y_true != i))
        FN = np.sum((y_pred != i) & (y_true == i))

        if TP + FP > 0:
            precision_macro[i] = TP / (TP + FP)
        else:
            precision_macro[i] = 0

        if TP + FN > 0:
            recall_macro[i] = TP / (TP + FN)
        else:
            recall_macro[i] = 0

        if precision_macro[i] + recall_macro[i] > 0:
            f1_macro[i] = 2 * precision_macro[i] * recall_macro[i] / (precision_macro[i] + recall_macro[i])
        else:
            f1_macro[i] = 0

    macro_precision = np.mean(precision_macro)
    macro_recall = np.mean(recall_macro)
    macro_f1 = np.mean(f1_macro)

    return macro_precision, macro_recall, macro_f1

# 主函数
def main():
    # 设置随机数种子
    seed_value = 42
    set_random_seeds(seed_value)

    csv_file = [
        'E:\libs\hammer-newlabel4\\20250311.csv',
        'E:\libs\hammer-newlabel4\\0626-2.csv',
        #'E:\libs\hammer-newlabel4\\20250702-1.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        'E:\libs\hammer-newlabel4\\20250704-2.csv',
        #'E:\libs\hammer-newlabel4\\20250904.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata\\train0.1.csv'
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv',
        #'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train1.0.csv'

    ]
    X, y = load_data_from_files(csv_file)

    guiyihua = "False"

    if guiyihua == "False":
        X_train, X_val, y_train_, y_val_ = train_test_split(X, y, test_size=0.001, random_state=seed_value)
    else:
        X_scaled = normalize_samples(X)
        X_train, X_val, y_train_, y_val_ = train_test_split(X_scaled, y, test_size=0.2, random_state=50)

    y_train = [[0 for j in range(17)] for i in range(len(y_train_))]
    for i in range(len(y_train_)):
        y_train[i][int(y_train_[i])] = 1
    y_val = [[0 for j in range(17)] for i in range(len(y_val_))]
    y_train = np.array(y_train, dtype=float)
    for i in range(len(y_val_)):
        y_val[i][int(y_val_[i])] = 1
    y_val = np.array(y_val, dtype=float)

    model = build_model((7062, 1), 17)
    #model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_val, y_val), callbacks=[reduce_lr,tensorboard])
    model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_val, y_val),
              callbacks=[reduce_lr])


    # 测试集评估
    X_test, y_test_ = load_data('E:\libs\hammer-newlabel4\\20250702-1.csv')
    
    if guiyihua == "False":
        pass
    else:
        X_test = normalize_samples(X_test)

    y_test = [[0 for j in range(17)] for i in range(len(y_test_))]
    for i in range(len(y_test_)):
        y_test[i][int(y_test_[i])] = 1
    y_test = np.array(y_test, dtype=float)


    #X_test=X_val
    #y_test=y_val


    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # 计算宏平均精确率、召回率和F1分数
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test_classes, y_pred_classes, 17)

    # 计算准确率
    accuracy = np.mean(y_test_classes == y_pred_classes)
    print("accuracy:",accuracy)
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

    # 绘制混淆矩阵
    #class_names = [str(i) for i in range(1, 18)]
    #plot_confusion_matrix(y_test_classes, y_pred_classes, class_names)

    # 测试集评估
    X_test, y_test_ = load_data('E:\libs\hammer-newlabel4\\transferlearningdata-new\\0904+1119_new_test.csv')

    if guiyihua == "False":
        pass
    else:
        X_test = normalize_samples(X_test)

    y_test = [[0 for j in range(17)] for i in range(len(y_test_))]
    for i in range(len(y_test_)):
        y_test[i][int(y_test_[i])] = 1
    y_test = np.array(y_test, dtype=float)

    # X_test=X_val
    # y_test=y_val

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # 计算宏平均精确率、召回率和F1分数
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test_classes, y_pred_classes, 17)

    # 计算准确率
    accuracy = np.mean(y_test_classes == y_pred_classes)
    print("accuracy:", accuracy)
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

if __name__ == '__main__':
    main()