import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
import keras_tuner as kt
import keras

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
# 创建回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

import os
import random as python_random

# 设置随机种子
seed_value = 42
np.random.seed(seed_value)
python_random.seed(seed_value)
tf.random.set_seed(seed_value)

# 禁用并行计算
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 确保 GPU 计算的确定性
tf.config.experimental.enable_op_determinism()

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
    #model.add(SelfAttention())
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
    plt.title(' ')
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




import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_loss_curve(h1):
    """
    绘制双Y轴曲线：左轴 Accuracy，右轴 Loss
    """
    # 1. 数据提取
    loss = h1.history['loss']
    accuracy = h1.history['accuracy']
    epochs_range = range(1, len(loss) + 1)

    # 2. 创建画布和左轴 (ax1) - 用于 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    color_acc = 'tab:red'  # 定义 Accuracy 的颜色 (红色)
    ax1.set_xlabel('Epochs',fontsize=15)
    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=15)
    # 绘制 Accuracy 曲线
    line1 = ax1.plot(epochs_range, accuracy, color=color_acc, linewidth=2, label='Training Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc,labelsize=15) # 让左侧刻度数字也变成红色
    ax1.tick_params(axis='x',  labelsize=15)
    ax1.grid(True, alpha=0.3)

    # 3. 创建右轴 (ax2) - 用于 Loss - 共享 X 轴
    ax2 = ax1.twinx()

    color_loss = 'tab:blue' # 定义 Loss 的颜色 (蓝色)
    ax2.set_ylabel('Loss', color=color_loss, fontsize=15)
    # 绘制 Loss 曲线
    line2 = ax2.plot(epochs_range, loss, color=color_loss, linewidth=2, label='Training Loss')
    ax2.tick_params(axis='y', labelcolor=color_loss,labelsize=15) # 让右侧刻度数字也变成蓝色

    # 4. 合并图例 (因为有两个轴，图例需要手动合并)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right',fontsize=15) # 将图例放在合适的位置

    #plt.title('Training Metrics: Accuracy vs Loss')
    plt.tight_layout() # 防止标签被切掉
    plt.show()



def build_hypermodel(hp):
    """
    使用Keras Tuner的超参数构建函数
    hp: HyperParameters对象
    """
    model = Sequential()

    # 搜索Conv1D的filters数量 (8, 16, 32)
    filters = hp.Int('filters', min_value=8, max_value=32, step=8)
    #filters = hp.Int('filters', min_value=16, max_value=16, step=8)
    # 搜索kernel_size (40, 80, 120)
    kernel_size = hp.Choice('kernel_size', values=[40, 80, 120])
    #kernel_size = hp.Choice('kernel_size', values=[40])

    # 搜索strides (10, 20, 40)
    strides = hp.Choice('strides', values=[10, 20, 40])
    #strides = hp.Choice('strides', values=[10])

    # 第一层Conv1D
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation='relu',
        input_shape=(7062, 1)
    ))

    # 搜索Dropout rate (0.1, 0.2, 0.3, 0.5)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    #dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.2, step=0.1)
    model.add(Dropout(dropout_rate))

    # 自注意力层（保持固定，也可以搜索参数）
    #model.add(SelfAttention())

    model.add(Flatten())

    # 搜索Dense层units (64, 128, 256)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    #dense_units = hp.Int('dense_units', min_value=256, max_value=256, step=64)
    model.add(Dense(dense_units, activation='relu'))

    # 输出层（固定17个类别）
    model.add(Dense(17, activation='softmax'))

    # 搜索学习率 (0.0001, 0.001, 0.01)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    #learning_rate = hp.Choice('learning_rate', values=[1e-3])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model


# 或者使用更简洁的基于sklearn的回调方式（推荐，更准确）
from sklearn.metrics import f1_score
import numpy as np


class MacroF1Callback(keras.callbacks.Callback):
    """在验证集上计算Macro F1并保存到logs中"""

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        # 计算macro f1
        macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

        # 添加到logs中，这样tuner可以追踪
        logs['val_macro_f1'] = macro_f1
        print(f' - val_macro_f1: {macro_f1:.4f}')





def main():
    seed_value = 42
    set_random_seeds(seed_value)

    # 1. 唯一数据源
    csv_file = [
        'E:\libs\hammer-newlabel4\\20250311.csv',
         'E:\libs\hammer-newlabel4\\0626-2.csv',
        # 'E:\libs\hammer-newlabel4\\20250702-1.csv',
         'E:\libs\hammer-newlabel4\\20250702-2.csv',
         'E:\libs\hammer-newlabel4\\20250703.csv',
         'E:\libs\hammer-newlabel4\\20250704-1.csv',
         'E:\libs\hammer-newlabel4\\20250704-2.csv',

    ]
    X, y = load_data_from_files(csv_file)
    test_csv = 'E:\libs\hammer-newlabel4\\20250702-1.csv'
    X_test, y_test_ = load_data(test_csv)
    # 2. 手动调这里：验证集占“剩余 80 %”的比例
    val_ratio = 0.2         # 20 %、30 %、40 % 随意改


    # 3. 拆分成train / val
    X_train, X_val, y_train_, y_val_ = train_test_split(
        X, y,
        test_size=val_ratio,          # 注意：这里是占 80 % 的比例
        random_state=seed_value,
        stratify=y)


    # one-hot
    num_cls = 17
    y_train = tf.keras.utils.to_categorical(y_train_, num_cls)
    y_val = tf.keras.utils.to_categorical(y_val_, num_cls)
    y_test = tf.keras.utils.to_categorical(y_test_, num_cls)

    # ===== Keras Tuner 超参数搜索 =====
    print("开始超参数搜索...")

    # 创建Hyperband搜索器（推荐，比RandomSearch更高效）
    tuner = kt.Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_macro_f1',direction="max"),  # 优化目标：验证集准确率  'val_macro_f1'  'val_accuracy'
        max_epochs=50,  # 每个模型最多训练50轮
        factor=3,  # 淘汰因子，每轮保留1/3模型
        directory='my_dir',  # 存储搜索结果的目录
        project_name='cnn_tuning',  # 项目名称
        overwrite=True
    )
    macro_f1_callback = MacroF1Callback(validation_data=(X_val, y_val))
    # 或者使用RandomSearch（更穷举，但慢）
    # tuner = kt.RandomSearch(
    #     build_hypermodel,
    #     objective='val_accuracy',
    #     max_trials=20,             # 尝试20种不同参数组合
    #     directory='my_dir',
    #     project_name='lcan_random_search'
    # )

    # 早停回调（防止单个模型训练过久）
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # 5轮不改善就停止
        restore_best_weights=True
    )

    # 学习率调度（可选）
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    # 执行搜索
    tuner.search(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[macro_f1_callback,stop_early, reduce_lr],
        verbose=2
    )

    # 获取最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n最佳超参数组合：")
    print(f"Filters: {best_hps.get('filters')}")
    print(f"Kernel Size: {best_hps.get('kernel_size')}")
    print(f"Strides: {best_hps.get('strides')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate'):.2f}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    # 使用最佳参数构建最终模型，并训练更充分
    print("\n使用最佳参数训练最终模型...")
    best_model = tuner.hypermodel.build(best_hps)

    # 训练最终模型（可以增加epochs）
    history = best_model.fit(
        X_train, y_train,
        epochs=20,  # 更多轮数
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr],
        verbose=1
    )

    # 后续评估代码保持不变...
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_test_classes)
    print(f"最终模型准确率: {accuracy:.4f}")

    # 保存模型
    best_model.save('best_lcan_model.h5')
    print("模型已保存为 'best_lcan_model.h5'")
    # 评估
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_test_classes)
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(
        y_test_classes, y_pred_classes, num_cls)

    print("accuracy:", accuracy)
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

    class_names = [str(i) for i in range(1, 18)]
    plot_confusion_matrix(y_test_classes, y_pred_classes, class_names)

if __name__ == '__main__':
    main()

