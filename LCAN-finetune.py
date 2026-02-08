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
    model.add(SelfAttention())
    model.add(Flatten())  # 或者使用 GlobalMaxPooling1D()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
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
    plt.title('LCAN transfer learning-5epoch')
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


# 冻结特定层
def freeze_layers(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            break
        layer.trainable = False

# 解冻特定层（可选）
def unfreeze_layers(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            break
        layer.trainable = True


def plot_two_stage_loss(h1, h2):
    """
    绘制两个阶段拼接的 Loss 和 Accuracy 曲线
    h1: 预训练阶段的 history
    h2: 微调阶段的 history
    """
    # 提取数据
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']

    # 阶段分界点
    stage1_end = len(h1.history['loss'])

    # 创建画布
    plt.figure(figsize=(12, 5))

    # 子图 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Val Loss', color='orange', linestyle='--')
    plt.axvline(x=stage1_end, color='green', linestyle=':', linewidth=2, label='Start Fine-tuning')

    plt.title('Loss Curve (Pre-train + Fine-tune)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Train Acc', color='blue')
    plt.plot(val_acc, label='Val Acc', color='orange', linestyle='--')
    plt.axvline(x=stage1_end, color='green', linestyle=':', linewidth=2, label='Start Fine-tuning')

    plt.title('Accuracy Curve (Pre-train + Fine-tune)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

'''
def plot_loss_curve(h1, h2):
    """
    绘制两个阶段拼接的 Loss 曲线
    """
    # 1. 数据拼接
    loss = h1.history['loss'] + h2.history['loss']
    accuracy= h1.history['accuracy'] + h2.history['accuracy']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    # 2. 获取第一阶段结束的位置（用于画竖线）
    stage1_end = len(h1.history['loss'])

    epochs_range = range(1, len(loss) + 1)
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range,loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, accuracy, label='Training Accuracy', color='red', linewidth=2)
    #plt.plot(epochs_range,val_loss, label='Validation Loss', color='red', linestyle='--', linewidth=2)

    # 添加竖线区分阶段
    #plt.axvline(x=stage1_end - 0.5, color='green', linestyle=':', linewidth=2, label='Start Transfer/Fine-tune')

    #plt.title('Loss Curve (Pre-training -> Fine-tuning)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss_curve(h1, h2):
    """
    绘制双阶段拼接曲线：左轴 Accuracy (红色)，右轴 Loss (蓝色)
    """
    # 1. 数据拼接 (Stage 1 + Stage 2)
    loss = h1.history['loss'] + h2.history['loss']
    accuracy = h1.history['accuracy'] + h2.history['accuracy']

    # 获取第一阶段结束的位置，用于画竖线
    stage1_end = len(h1.history['loss'])
    # 横坐标从 1 开始
    epochs_range = range(1, len(loss) + 1)

    # 2. 创建画布和左侧坐标轴 (ax1) -> 用于 Accuracy
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- 设置 X 轴 ---
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 强制 X 轴只显示整数
    ax1.tick_params(axis='x', labelsize=15)  # X 轴刻度数字大小

    # --- 设置左轴 (Accuracy) ---
    color_acc = 'red'
    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=15)
    line1 = ax1.plot(epochs_range, accuracy, label='Training Accuracy', color=color_acc, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=15)  # 左轴刻度数字大小及颜色
    ax1.grid(True, alpha=0.3)  # 网格线

    # 3. 创建右侧坐标轴 (ax2) -> 用于 Loss
    ax2 = ax1.twinx()

    color_loss = 'blue'
    ax2.set_ylabel('Loss', color=color_loss, fontsize=15)
    line2 = ax2.plot(epochs_range, loss, label='Training Loss', color=color_loss, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=15)  # 右轴刻度数字大小及颜色

    # 4. 添加阶段分割线 (预训练 vs 微调)
    plt.axvline(x=stage1_end, color='green', linestyle=':', linewidth=2, label='Start Fine-tuning')

    # 5. 合并图例
    # 因为有两个坐标轴，需要手动把它们的线和标签取出来放在一起
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    #ax1.legend(lines, labels, loc='center right', fontsize=15)

    #plt.title('Training Metrics (Pre-training + Fine-tuning)', fontsize=14)
    plt.tight_layout()
    plt.show()



# 主函数
def main():
    # 设置随机数种子
    seed_value = 42
    set_random_seeds(seed_value)

    # 加载原始训练集
    train_file_list = [
        'E:\libs\hammer-newlabel4\\20250311.csv',
        'E:\libs\hammer-newlabel4\\0626-2.csv',
        #'E:\libs\hammer-newlabel4\\20250702-1.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        'E:\libs\hammer-newlabel4\\20250704-2.csv',
        #'E:\libs\hammer-newlabel4\\20250904.csv'
        #'E:\libs\hammer-newlabel4\\transferlearningdata/train0.9.csv'
    ]
    X, y_ = load_data_from_files(train_file_list)
    X_train,X_val,y_train_,y_val_ = train_test_split(X,y_,test_size=0.001,random_state=seed_value)


    trainval_file = 'E:\libs\hammer-newlabel4\\transferlearningdata-new/train0.2.csv'
    X_trainval_transfer, y_trainval_transfer_ = load_data(trainval_file)

    X_train_transfer, X_val_transfer, y_train_transfer_, y_val_transfer_ = train_test_split(
        X_trainval_transfer, y_trainval_transfer_, test_size=0.001, random_state=seed_value
    )

    # 将标签转换为 one-hot 编码
    num_labels = 17
    y_train = np.zeros((len(y_train_), num_labels))
    for i in range(len(y_train_)):
        y_train[i][int(y_train_[i])] = 1
    y_val = np.zeros((len(y_val_), num_labels))
    for i in range(len(y_val_)):
        y_val[i][int(y_val_[i])] = 1
    y_train_transfer = np.zeros((len(y_train_transfer_), num_labels))
    for i in range(len(y_train_transfer_)):
        y_train_transfer[i][int(y_train_transfer_[i])] = 1
    y_val_transfer = np.zeros((len(y_val_transfer_), num_labels))
    for i in range(len(y_val_transfer_)):
        y_val_transfer[i][int(y_val_transfer_[i])] = 1


    
    # 构建模型
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape, num_labels)

    # 先用原始训练集训练模型20个epoch
    h1=model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=8, validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
              callbacks=[reduce_lr])
    '''
    # 冻结特定层
    model = model0.layers[:5]
    model = tf.keras.Sequential(model)
    #model.build((None, 28, 28, 1))

    # Setting trainable = False will make them non-trainable.
    model.trainable = False
    model.add(tf.keras.layers.Dense(17, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # 检查每一层的 trainable 属性
    for layer in model.layers:
        print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

    '''
    #input_shape = (X_train.shape[1], 1)
    #model = build_model(input_shape, num_labels)
    # 再用测试集文件的70%做迁移学习训练集，训练5个epoch
    # 重新设置学习率为0.001
    model.optimizer.learning_rate.assign(0.001)
    h2=model.fit(X_train_transfer.reshape(-1, X_train_transfer.shape[1], 1), y_train_transfer, epochs=20, batch_size=8, validation_data=(X_val_transfer.reshape(-1, X_val_transfer.shape[1], 1), y_val_transfer),
              callbacks=[reduce_lr])
    plot_two_stage_loss(h1,h2)
    plot_loss_curve(h1,h2)

    test_file = 'E:\libs\hammer-newlabel4\\0904+1119_test.csv'
    X_test_transfer, y_test_transfer_ = load_data(test_file)

    y_test_transfer = np.zeros((len(y_test_transfer_), num_labels))
    for i in range(len(y_test_transfer_)):
        y_test_transfer[i][int(y_test_transfer_[i])] = 1

    y_pred_prob = model.predict(X_test_transfer.reshape(-1, X_test_transfer.shape[1], 1))
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_transfer_classes = np.argmax(y_test_transfer, axis=1)

    # 计算宏平均精确率、召回率和F1分数
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test_transfer_classes, y_pred_classes, 17)
    accuracy = np.mean(y_test_transfer_classes == y_pred_classes)
    print(" target accuracy:", accuracy)
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

    # 绘制混淆矩阵
    num_classes = len(np.unique(y_train_))
    class_names = [str(i) for i in range(1, num_classes + 1)]
    #plot_confusion_matrix(y_test_transfer_classes, y_pred_classes, class_names)

    test_file = 'E:\libs\hammer-newlabel4\\20250702-1.csv'
    X_test_transfer, y_test_transfer_ = load_data(test_file)

    y_test_transfer = np.zeros((len(y_test_transfer_), num_labels))
    for i in range(len(y_test_transfer_)):
        y_test_transfer[i][int(y_test_transfer_[i])] = 1


    y_pred_prob = model.predict(X_test_transfer.reshape(-1, X_test_transfer.shape[1], 1))
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_transfer_classes = np.argmax(y_test_transfer, axis=1)

    # 计算宏平均精确率、召回率和F1分数
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test_transfer_classes, y_pred_classes, 17)
    accuracy = np.mean(y_test_transfer_classes == y_pred_classes)
    print("source accuracy:", accuracy)
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

if __name__ == '__main__':
    main()