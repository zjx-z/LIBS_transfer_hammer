import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import random as python_random
import os

# 设置随机种子（保持可重复性）
seed_value = 42
np.random.seed(seed_value)
python_random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['OMP_NUM_THREADS'] = '1'


# ===== 数据加载函数（保持不变） =====
def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1].values
    return X, y


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


# ===== 修改1: 使用Keras Tuner的模型构建函数 =====
def build_hypermodel(hp):
    """
    使用Keras Tuner的超参数构建MLP模型
    hp: HyperParameters对象
    """
    model = Sequential()

    # 搜索第一层Dense的神经元数量 (64, 128, 256, 512)
    # 对于LIBS数据(7062维)，通常需要较大容量
    units_1 = hp.Int('units_1', min_value=128, max_value=512, step=128)
    #units_1 = hp.Int('units_1', min_value=384, max_value=384, step=128)
    # 第一层（必须匹配输入维度7062）
    model.add(Dense(units_1, activation='relu', input_shape=(7062,)))

    # 搜索Dropout率 (0.1 - 0.5)
    dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    #dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.1, step=0.1)
    model.add(Dropout(dropout_1))

    # 搜索是否添加第二层Dense（可选）
    if hp.Boolean('use_second_layer', default=True):
        # 搜索第二层神经元数量（必须小于第一层，防止瓶颈）
        units_2 = hp.Int('units_2', min_value=64, max_value=256, step=64)
        model.add(Dense(units_2, activation='relu'))

        # 第二层Dropout（可选，可以与第一层不同或共享）
        dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_2))
        #pass


    # 搜索是否添加第三层Dense（更深网络）
    if hp.Boolean('use_third_layer', default=False):
        units_3 = hp.Int('units_3', min_value=32, max_value=128, step=32)
        model.add(Dense(units_3, activation='relu'))
        model.add(Dropout(0.2))  # 固定较小的dropout
        #pass

    # 输出层（固定17个类别）
    model.add(Dense(17, activation='softmax'))

    # 搜索优化器类型
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    #optimizer_choice = hp.Choice('optimizer', values=['rmsprop'])

    # 搜索学习率 (对数尺度，从0.0001到0.01)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    #learning_rate=hp.Float('learning_rate',min_value=0.000136,max_value=0.000136)

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


# ===== 评估指标计算（保持不变） =====
def calculate_macro_metrics(y_true, y_pred, num_classes):
    precision_macro = np.zeros(num_classes)
    recall_macro = np.zeros(num_classes)
    f1_macro = np.zeros(num_classes)

    for i in range(num_classes):
        TP = np.sum((y_pred == i) & (y_true == i))
        FP = np.sum((y_pred == i) & (y_true != i))
        FN = np.sum((y_pred != i) & (y_true == i))

        precision_macro[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_macro[i] = TP / (TP + FN) if (TP + FN) > 0 else 0

        if precision_macro[i] + recall_macro[i] > 0:
            f1_macro[i] = 2 * precision_macro[i] * recall_macro[i] / (precision_macro[i] + recall_macro[i])
        else:
            f1_macro[i] = 0

    return np.mean(precision_macro), np.mean(recall_macro), np.mean(f1_macro)


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




# ===== 修改2: 主函数包含超参数搜索 =====
def main():
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

    # 3. 第一次拆分：先拿出 20 % 做测试集

    X_train, X_val, y_train_, y_val_ = train_test_split(
        X, y,
        test_size=val_ratio,          # 注意：这里是占 80 % 的比例
        random_state=seed_value,
        stratify=y)


    # 3. One-hot编码（保持不变）
    num_cls = 17
    y_train = tf.keras.utils.to_categorical(y_train_, num_cls)
    y_val = tf.keras.utils.to_categorical(y_val_, num_cls)
    y_test = tf.keras.utils.to_categorical(y_test_, num_cls)

    print(f"训练集形状: {X_train.shape}")
    print(f"输入特征维度: {X_train.shape[1]}")

    # 4. 创建Keras Tuner（推荐Hyperband算法）
    tuner = kt.Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_macro_f1',direction="max"),  # 优化目标：验证集准确率  val_macro_f1  val_accuracy
        max_epochs=50,  # 每个模型最多训练50轮
        factor=3,  # 淘汰因子（每轮保留1/3）
        directory='mlp_tuning',  # 存储目录
        project_name='libs_mlp_f1_optimization',  # 项目名称
        seed=seed_value,  # 保证可重复性
        overwrite=True
    )
    #训练时使用回调计算macro f1
    macro_f1_callback=MacroF1Callback(validation_data=(X_val,y_val))
    # 可选：使用RandomSearch（更穷举但慢）
    # tuner = kt.RandomSearch(
    #     build_hypermodel,
    #     objective='val_accuracy',
    #     max_trials=30,                # 尝试30种不同架构
    #     directory='mlp_tuning',
    #     project_name='libs_mlp_random'
    # )

    # 5. 早停回调（防止单个配置训练过久）
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # 10轮不改善就停止
        restore_best_weights=True
    )

    # 学习率调度（可选）
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    # 6. 执行超参数搜索
    print("\n" + "=" * 50)
    print("开始超参数搜索...")
    print("=" * 50)

    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[macro_f1_callback,stop_early, reduce_lr],
        verbose=1
    )

    # 7. 获取最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n" + "=" * 50)
    print("最佳超参数组合：")
    print("=" * 50)
    print(f"第一层神经元: {best_hps.get('units_1')}")
    print(f"第一层Dropout: {best_hps.get('dropout_1'):.2f}")

    if best_hps.get('use_second_layer'):
        print(f"第二层神经元: {best_hps.get('units_2')}")
        print(f"第二层Dropout: {best_hps.get('dropout_2'):.2f}")
    else:
        print("第二层: 未使用")

    if best_hps.get('use_third_layer'):
        print(f"第三层神经元: {best_hps.get('units_3')}")
    else:
        print("第三层: 未使用")

    print(f"优化器: {best_hps.get('optimizer')}")
    print(f"学习率: {best_hps.get('learning_rate'):.6f}")

    # 8. 使用最佳参数训练最终模型（更多轮数）
    print("\n使用最佳参数训练最终模型...")
    best_model = tuner.hypermodel.build(best_hps)

    # 重新训练更充分（可以增加epochs）
    history = best_model.fit(
        X_train, y_train,
        epochs=50,  # 更多轮数
        batch_size=32,  # MLP通常可以用更大batch_size（可选参数搜索）
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr],
        verbose=1
    )

    # 9. 评估最终模型
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_test_classes)
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(
        y_test_classes, y_pred_classes, num_cls)

    print("\n" + "=" * 50)
    print("最终模型性能：")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

    # 10. 保存模型
    best_model.save('best_mlp_model.h5')
    print("\n模型已保存为 'best_mlp_model.h5'")

    # 可选：保存最佳参数到文本
    with open('best_hyperparameters.txt', 'w') as f:
        for param, value in best_hps.values.items():
            f.write(f"{param}: {value}\n")


if __name__ == '__main__':
    main()