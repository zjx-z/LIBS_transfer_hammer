import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Dropout, Input
from keras.layers import Conv1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K
import random as python_random
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import winsound

# 设置随机种子
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# 创建回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)


# 加载数据函数
def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1].values#这里-1后面加个冒号就跑不通了
    return X, y


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values#这里-1后面加个冒号就跑不通了
    return X, y


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


def mmd_loss(src_feat, tgt_feat):
    src_mean = K.mean(src_feat, axis=0)
    tgt_mean = K.mean(tgt_feat, axis=0)
    return K.sum(K.square(src_mean - tgt_mean))


def build_mmd_model(input_shape, num_labels, lambda_mmd=1.0):
    """彻底重构：不使用任何嵌套 Model，直接共享层"""
    #单域用Sequential装层。双域的没有装起来
    # 1. 定义共享层（在函数外部创建，确保唯一性）
    conv_layer = Conv1D(8, 80, strides=20, activation='relu')
    dropout_layer = Dropout(0.2)
    attention_layer = SelfAttention()
    flatten_layer = Flatten()
    dense_layer = Dense(128, activation='relu', name='feat')
    classifier1 = Dense(num_labels, activation='softmax', name='cls1')
    classifier2 = Dense(num_labels, activation='softmax', name='cls2')

    # 2. 定义输入
    src_in = Input(shape=input_shape, name='src_input')
    tgt_in = Input(shape=input_shape, name='tgt_input')

    # 3. 源域前向传播
    x_src = conv_layer(src_in)
    x_src = dropout_layer(x_src)
    x_src = attention_layer(x_src)
    x_src = flatten_layer(x_src)
    feat_src = dense_layer(x_src)
    logits_src = classifier1(feat_src)

    # 4. 目标域前向传播（自动共享权重）
    x_tgt = conv_layer(tgt_in)
    x_tgt = dropout_layer(x_tgt)
    x_tgt = attention_layer(x_tgt)
    x_tgt = flatten_layer(x_tgt)
    feat_tgt = dense_layer(x_tgt)
    logits_tgt = classifier2(feat_tgt)

    # 5. 构建模型
    model = Model(inputs=[src_in, tgt_in], outputs=[logits_src, logits_tgt])

    # 6. 添加MMD损失（此时 feat_src/tgt 是纯张量，无嵌套结构）
    mmd = mmd_loss(feat_src, feat_tgt)
    model.add_loss(lambda_mmd * mmd)

    # 7. 编译
    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy'],#这里把源域交叉熵损失算上了
        loss_weights=[1, 1],
        optimizer=Adam(1e-4),
        metrics=['accuracy']
    )
    feature_ectractor = Model(inputs=[src_in, tgt_in], outputs=[feat_src, feat_tgt])
    return model,feature_ectractor


def data_generator(X_s, y_s, X_t, y_t, batch_size=16):
    """生成器：同时提供源域和目标域数据及标签"""#比单域的输出多一个[y_s[ind_s]
    while True:
        ind_s = np.random.choice(len(X_s), 3*batch_size)
        ind_t = np.random.choice(len(X_t), batch_size)
        yield ([X_s[ind_s], X_t[ind_t]], [y_s[ind_s], y_t[ind_t]])


def calculate_macro_metrics(y_true, y_pred, num_classes):
    precision_macro = np.zeros(num_classes)
    recall_macro = np.zeros(num_classes)
    f1_macro = np.zeros(num_classes)

    for i in range(num_classes):
        TP = np.sum((y_pred == i) & (y_true == i))
        FP = np.sum((y_pred == i) & (y_true != i))
        FN = np.sum((y_pred != i) & (y_true == i))

        precision_macro[i] = TP / (TP + FP) if TP + FP > 0 else 0
        recall_macro[i] = TP / (TP + FN) if TP + FN > 0 else 0
        f1_macro[i] = 2 * precision_macro[i] * recall_macro[i] / (precision_macro[i] + recall_macro[i]) if \
        precision_macro[i] + recall_macro[i] > 0 else 0

    return np.mean(precision_macro), np.mean(recall_macro), np.mean(f1_macro)

from sklearn.metrics import f1_score
def calculate_macro_f1(y_true, y_pred, num_classes):
    # 计算微平均 F1 Score
    macro_f1 = f1_score(y_true, y_pred, labels=np.arange(num_classes), average='macro')
    return macro_f1



import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE
def draw_tsne4(X_src_test, X_tgt_test, y_src_test, y_tgt_test, title, output_file):
    # 定义类别到颜色的映射
    color_map = {
        0: 'blue',  # 类别 0 使用蓝色
        1: 'blue',  # 类别 1 使用红色
        2: 'blue',  # 类别 2 使用绿色
        3: 'blue',  # 类别 0 使用蓝色
        4: 'blue',  # 类别 1 使用红色
        5: 'red',  # 类别 2 使用绿色
        6: 'red',  # 类别 0 使用蓝色
        7: 'red',  # 类别 1 使用红色
        8: 'red',  # 类别 2 使用绿色
        9: 'green',  # 类别 0 使用蓝色
        10: 'green',  # 类别 1 使用红色
        11: 'green',  # 类别 2 使用绿色
        12: 'green',  # 类别 0 使用蓝色
        13: 'green',  # 类别 1 使用红色
        14: 'yellow',  # 类别 2 使用绿色
        15: 'yellow',  # 类别 0 使用蓝色
        16: 'yellow',  # 类别 1 使用红色
    }

    # 展平特征向量
    X_src_test_2d = X_src_test
    X_tgt_test_2d = X_tgt_test

    # 合并展平后的特征
    all_feats = np.vstack((X_src_test_2d, X_tgt_test_2d))
    tsne = TSNE(n_components=3, random_state=42)  # 修改为3维
    all_feats_3d = tsne.fit_transform(all_feats)

    # 分离源域和目标域的特征
    src_feats_3d = all_feats_3d[:len(X_src_test_2d)]
    tgt_feats_3d = all_feats_3d[len(X_src_test_2d):]

    # 创建 Plotly 图形
    fig = go.Figure()

    # 添加源域数据
    for label in np.unique(y_src_test):
        indices = np.where(y_src_test == label)[0]
        fig.add_trace(
            go.Scatter3d(
                x=src_feats_3d[indices, 0],
                y=src_feats_3d[indices, 1],
                z=src_feats_3d[indices, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_map[label],
                    line=dict(width=1, color=color_map[label]),
                    symbol='circle-open'
                ),
                name=f'Source Domain - Class {label}'
            )
        )

    # 添加目标域数据
    for label in np.unique(y_tgt_test):
        indices = np.where(y_tgt_test == label)[0]
        fig.add_trace(
            go.Scatter3d(
                x=tgt_feats_3d[indices, 0],
                y=tgt_feats_3d[indices, 1],
                z=tgt_feats_3d[indices, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=color_map[label],
                    line=dict(width=0.3, color=color_map[label]),
                    symbol='x'
                ),
                name=f'Target Domain - Class {label}'
            )
        )

    # 设置标题
    fig.update_layout(title=title)

    # 保存为 HTML 文件
    fig.write_html(output_file)

    # 显示图形
    fig.show()






def main():
    seed_value = 40
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # 加载源域（训练）
    src_files = [
        'E:\\libs\\hammer-newlabel4\\20250311.csv',
        'E:\\libs\\hammer-newlabel4\\0626-2.csv',
        'E:\\libs\\hammer-newlabel4\\20250702-2.csv',
        'E:\\libs\\hammer-newlabel4\\20250703.csv',
        'E:\\libs\\hammer-newlabel4\\20250704-1.csv',
        'E:\\libs\\hammer-newlabel4\\20250704-2.csv'
    ]
    X_src, y_src_ = load_data_from_files(src_files)
    y_src = np.eye(17)[y_src_.astype(int)]

    # 加载目标域（训练）
    tgt_train_file = 'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train1.0.csv'
    X_tgt_train, y_tgt_train_ = load_data(tgt_train_file)
    y_tgt_train = np.eye(17)[y_tgt_train_.astype(int)]
    tgt_val_file = 'E:\libs\hammer-newlabel4\\transferlearningdata/val0.2.csv'
    X_tgt_val, y_tgt_val_ = load_data(tgt_val_file)
    y_tgt_val = np.eye(17)[y_tgt_val_.astype(int)]
    tgt_test_file = 'E:\libs\hammer-newlabel4\\transferlearningdata-new\\0904+1119_new_test.csv'
    X_tgt_test, y_tgt_test_ = load_data(tgt_test_file)
    y_tgt_test = np.eye(17)[y_tgt_test_.astype(int)]
    # 加载源域测试集（用于评估防遗忘效果）
    src_test_file = 'E:\\libs\\hammer-newlabel4\\20250702-1.csv'
    X_src_test, y_src_test_ = load_data(src_test_file)
    y_src_test = np.eye(17)[y_src_test_.astype(int)]

    # 目标函数
    def objective(trial):
        lambda_mmd = trial.suggest_float('lambda_mmd', 0.01, 10.0, log=True)
        model,feature_extractor = build_mmd_model((7062, 1), 17, lambda_mmd=lambda_mmd)
        steps = max(len(X_src), len(X_tgt_train)) // 16
        model.fit(
            data_generator(X_src, y_src, X_tgt_train, y_tgt_train, 16),
            steps_per_epoch=steps,
            epochs=10,
            validation_data=([X_src_test[:len(X_tgt_train)], X_tgt_train],
                             [y_src_test[:len(y_tgt_train)], y_tgt_train]),#这里也加上了[y_src_test[:len(y_tgt_test)]
            verbose=0)
        # 评估目标域性能
        dummy_src = np.zeros_like(X_tgt_train)
        eval_results = model.evaluate([dummy_src, X_tgt_train],
                                    [y_src_test[:len(y_tgt_train)], y_tgt_train], verbose=0)
        print(eval_results)
        print(model.metrics_names)

        '''
        #eval_results[0]: 总损失，包括源域损失、目标域损失和 MMD 损失的加权和。
        #eval_results[1]: 源域的损失（categorical_crossentropy）。
        #eval_results[2]: 目标域的损失（categorical_crossentropy）。
        #eval_results[1]: 源域的
        '''

        return eval_results[3]+eval_results[4]
    # Optuna调参
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=1, n_jobs=1)

    best_lambda = study.best_params['lambda_mmd']
    print("Best λ_mmd:", best_lambda)

    # 最终训练
    #best_model = build_mmd_model((7062, 1), 17, lambda_mmd=best_lambda)
    best_model,feature_extractor = build_mmd_model((7062, 1), 17, lambda_mmd=0.1)
    steps = max(len(X_src), len(X_tgt_train)) // 16
    best_model.fit(
        data_generator(X_src, y_src, X_tgt_train, y_tgt_train, 16),
        steps_per_epoch=steps,
        epochs=20,
        validation_data=([X_src_test[:len(X_tgt_train)], X_tgt_train],
                         [y_src_test[:len(y_tgt_train)], y_tgt_train]),
        callbacks=[reduce_lr]
    )

    # 目标域评估
    dummy_src = np.zeros_like(X_tgt_test)
    y_pred_tgt = best_model.predict([dummy_src, X_tgt_test])[1]
    acc_tgt = accuracy_score(np.argmax(y_tgt_test, 1), np.argmax(y_pred_tgt, 1))
    macro_f1=calculate_macro_f1(np.argmax(y_tgt_test, 1), np.argmax(y_pred_tgt, 1),17)
    print(f"Target domain accuracy: {acc_tgt:.4f}")
    print(f"Target domain macro_f1: {macro_f1:.4f}")

    # 源域评估（验证防遗忘效果）
    y_pred_src = best_model.predict([X_src_test, X_src_test])[0]
    acc_src = accuracy_score(np.argmax(y_src_test, 1), np.argmax(y_pred_src, 1))
    macro_f1 = calculate_macro_f1(np.argmax(y_src_test, 1), np.argmax(y_pred_src, 1),17)
    print(f"Source domain accuracy: {acc_src:.4f}")
    print(f"Source domain macro_F1: {macro_f1:.4f}")

    feat_src_test = feature_extractor([X_src_test, X_src_test])[0]
    feat_tgt_test = feature_extractor([X_tgt_test, X_tgt_test])[1]
    draw_tsne4(feat_src_test, feat_tgt_test, y_src_test_, y_tgt_test_, "Embedding", "MMD1.html")

    # 混淆矩阵（目标域）
    conf_matrix = confusion_matrix(np.argmax(y_tgt_test, 1), np.argmax(y_pred_tgt, 1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Target Domain')
    #plt.show()

    winsound.Beep(2500, 1000)




if __name__ == '__main__':
    main()