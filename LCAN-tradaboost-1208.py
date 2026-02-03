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
from keras.models import Model
import time
import winsound

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
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def calculate_macro_metrics(y_true, y_pred, num_classes):
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

# TrAdaBoost 实现
class TrAdaBoost:
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.beta = []

    def fit(self, X_source, y_source, X_target, y_target):
        print(X_source.shape)
        print(X_target.shape)
        print(y_source.shape)
        print(y_target.shape)
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        # 确保 y_source 和 y_target 是二维数组
        #y_source = y_source.reshape(-1, 1)
        #y_target = y_target.reshape(-1, 1)


        # 合并源域和目标域数据
        X = np.vstack((X_source, X_target))
        y = np.vstack((y_source, y_target))  # 使用 np.vstack 合并 y

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        # 初始化权重
        weights_source = np.ones(n_source) / n_source
        weights_target = np.ones(n_target) / n_target

        for i in range(self.n_estimators):
            # 合并权重
            weights = np.hstack((weights_source, weights_target))

            # 训练基学习器
            model = self.base_estimator()
            model.fit(X, y, sample_weight=weights)  # y需要是一维数组

            # 预测目标域数据
            y_pred = model.predict(X_target)
            error = np.mean(np.argmax(y_pred, axis=1) != np.argmax(y_target, axis=1))

            # 计算 beta
            beta = error / (1.0 - error + 1e-10)
            self.beta.append(beta)

            # 更新权重
            for j in range(n_source):
                if np.argmax(model.predict(X_source[j:j + 1]), axis=1) != np.argmax(y_source[j:j + 1], axis=1):
                    weights_source[j] *= beta ** self.learning_rate

            for j in range(n_target):
                if np.argmax(y_pred[j:j + 1], axis=1) != np.argmax(y_target[j:j + 1], axis=1):
                    weights_target[j] *= 1.0 / beta ** self.learning_rate

            # 归一化权重
            weights_source /= np.sum(weights_source)
            weights_target /= np.sum(weights_target)

            self.models.append(model)

    def predict(self, X):
        n_samples = X.shape[0]
        n_models = len(self.models)
        predictions = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.models):
            predictions[:, i] = np.argmax(model.predict(X),axis=1)
        final_pred = np.zeros(n_samples)
        for i in range(n_samples):
            # 为每个类别计算投票得分
            class_votes = np.zeros(17)  # 假设有 17 个类别
            for j in range(n_models):
                class_votes[int(predictions[i, j])] += np.log(1.0 / (self.beta[j] + 1e-10))
            final_pred[i] = np.argmax(class_votes)
        return final_pred.astype(int)


from sklearn.metrics import f1_score

def calculate_weighted_f1(y_true, y_pred, num_classes):
    # 计算每个类别的 F1 Score
    f1_scores = f1_score(y_true, y_pred, labels=np.arange(num_classes), average=None)
    # 计算每个类别的样本数量
    class_counts = np.bincount(y_true, minlength=num_classes)
    # 计算加权平均 F1 Score
    weighted_f1 = np.sum(f1_scores * class_counts) / np.sum(class_counts)
    return weighted_f1

def calculate_micro_f1(y_true, y_pred, num_classes):
    # 计算微平均 F1 Score
    micro_f1 = f1_score(y_true, y_pred, labels=np.arange(num_classes), average='micro')
    return micro_f1

def calculate_macro_f1(y_true, y_pred, num_classes):
    # 计算微平均 F1 Score
    macro_f1 = f1_score(y_true, y_pred, labels=np.arange(num_classes), average='macro')
    return macro_f1


# 主函数
def main():
    # 设置随机数种子
    seed_value = 40
    set_random_seeds(seed_value)

    # 加载源域数据
    csv_files = [
        'E:\\libs\\hammer-newlabel4\\20250311.csv',
        'E:\\libs\\hammer-newlabel4\\0626-2.csv',
        'E:\\libs\\hammer-newlabel4\\20250702-2.csv',
        'E:\\libs\\hammer-newlabel4\\20250703.csv',
        'E:\\libs\\hammer-newlabel4\\20250704-1.csv',
        'E:\\libs\\hammer-newlabel4\\20250704-2.csv'
    ]
    X_src, y_src = load_data_from_files(csv_files)

    # 加载目标域数据（训练集和测试集）
    X_tgt_train, y_tgt_train = load_data('E:\libs\hammer-newlabel4\\transferlearningdata-new\\train0.8.csv')
    X_tgt_test, y_tgt_test = load_data('E:\\libs\\hammer-newlabel4\\transferlearningdata-new\\0904+1119_new_test.csv')

    src_test_file = 'E:\\libs\\hammer-newlabel4\\20250702-1.csv'
    X_src_test, y_src_test_ = load_data(src_test_file)

    # 归一化数据
    #scaler = StandardScaler()
    #X_src = scaler.fit_transform(X_src)
    #X_tgt_train = scaler.transform(X_tgt_train)
    #X_tgt_test = scaler.transform(X_tgt_test)

    # 将数据转换为适合模型的格式
    X_src = X_src.reshape(X_src.shape[0], X_src.shape[1], 1)
    X_tgt_train = X_tgt_train.reshape(X_tgt_train.shape[0], X_tgt_train.shape[1], 1)
    X_tgt_test = X_tgt_test.reshape(X_tgt_test.shape[0], X_tgt_test.shape[1], 1)
    X_src_test=X_src_test.reshape(X_src_test.shape[0],X_src_test.shape[1],1)


    # 将标签转换为 one-hot 编码
    y_src = tf.keras.utils.to_categorical(y_src, num_classes=17)
    y_tgt_train = tf.keras.utils.to_categorical(y_tgt_train, num_classes=17)
    y_tgt_test = tf.keras.utils.to_categorical(y_tgt_test, num_classes=17)
    y_src_test=tf.keras.utils.to_categorical(y_src_test_,num_classes=17)

    # 确定较大和较小的数据集
    if X_src.shape[0] < X_tgt_train.shape[0]:
        print(1)
        smaller_X, smaller_y = X_src, y_src
        larger_X, larger_y = X_tgt_train, y_tgt_train
    else:
        print(2)
        smaller_X, smaller_y = X_tgt_train, y_tgt_train
        larger_X, larger_y = X_src, y_src

    # 重复较小的数据集
    repeat_factor = int(np.ceil(larger_X.shape[0] / smaller_X.shape[0]))
    smaller_X_repeated = np.tile(smaller_X, (repeat_factor, 1, 1))
    smaller_y_repeated = np.tile(smaller_y, (repeat_factor, 1))

    # 裁剪重复后的数据集，使其大小与较大数据集一致
    smaller_X_repeated = smaller_X_repeated[:larger_X.shape[0]]
    smaller_y_repeated = smaller_y_repeated[:larger_X.shape[0]]

    # 更新源域和目标域数据
    if X_src.shape[0] < X_tgt_train.shape[0]:
        print(3)
        X_src, y_src = smaller_X_repeated, smaller_y_repeated
    else:
        print(4)
        X_tgt_train, y_tgt_train = smaller_X_repeated, smaller_y_repeated





    # 构建模型
    #model = build_model((X_src.shape[1], 1), 17)

    #print(len(X_src),len(y_src))
    # 训练模型
    #model.fit(X_src, y_src, epochs=1, batch_size=8, validation_split=0.2, callbacks=[reduce_lr])

    # 使用 TrAdaBoost 调整源域样本权重
    def base_estimator():
        model = build_model((X_src.shape[1], 1), 17)
        return model



    print(len(X_src),len(y_src))
    tradaboost = TrAdaBoost(base_estimator, n_estimators=5, learning_rate=1.0)
    tradaboost.fit(X_src, y_src, X_tgt_train, y_tgt_train)

    # 使用 TrAdaBoost 的权重继续训练模型
    #weights = tradaboost.beta

    # 确保 sample_weight 是 NumPy 数组
    #if not isinstance(weights, np.ndarray):
    #    print("weights not numpy")
    #    weights = np.array(weights)

    # 确保 X_src 和 y_src 是 NumPy 数组
    #if not isinstance(X_src, np.ndarray):
    #    print("X_src not numpy")
    #    X_src = np.array(X_src)
    #if not isinstance(y_src, np.ndarray):
    #    print("y_src not numpy")
    #    y_src = np.array(y_src)


    #model.fit(X_src, y_src, epochs=1, batch_size=8, validation_split=0.2, callbacks=[reduce_lr], sample_weight=weights)

    # 在目标域测试集上评估模型
    y_tgt_pred_classes = tradaboost.predict(X_tgt_test)
    #y_tgt_pred_classes = np.argmax(y_tgt_pred, axis=1)
    y_tgt_test_classes = np.argmax(y_tgt_test, axis=1)

    accuracy = np.mean(y_tgt_test_classes == y_tgt_pred_classes)
    print("Target accuracy:", accuracy)
    macro_f1 = calculate_macro_f1(y_tgt_test_classes, y_tgt_pred_classes, num_classes=17)
    print(f"Macro-average F1 Score: {macro_f1:.4f}")
    weighted_f1_tgt = calculate_weighted_f1(y_tgt_test_classes, y_tgt_pred_classes, num_classes=17)
    print(f"Target domain weighted F1 Score: {weighted_f1_tgt:.4f}")
    micro_f1_tgt = calculate_micro_f1(y_tgt_test_classes, y_tgt_pred_classes, num_classes=17)
    print(f"Target domain micro F1 Score: {micro_f1_tgt:.4f}")

    # 在源域测试集上评估模型
    y_src_pred_classes = tradaboost.predict(X_src_test)
    # y_tgt_pred_classes = np.argmax(y_tgt_pred, axis=1)
    y_src_test_classes = np.argmax(y_src_test, axis=1)

    accuracy = np.mean(y_src_test_classes == y_src_pred_classes)
    print("Source accuracy:", accuracy)
    macro_f1 = calculate_macro_f1(y_src_test_classes, y_src_pred_classes, num_classes=17)
    print(f"Macro-average F1 Score: {macro_f1:.4f}")
    weighted_f1_tgt = calculate_weighted_f1(y_src_test_classes, y_src_pred_classes, num_classes=17)
    print(f"Target domain weighted F1 Score: {weighted_f1_tgt:.4f}")
    micro_f1_tgt = calculate_micro_f1(y_src_test_classes, y_src_pred_classes, num_classes=17)
    print(f"Target domain micro F1 Score: {micro_f1_tgt:.4f}")





if __name__ == '__main__':
    starttime = time.time()
    main()
    endtime = time.time()
    print(endtime - starttime)

    winsound.Beep(2500, 1000)