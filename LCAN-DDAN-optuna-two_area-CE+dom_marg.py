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
import time

# 设置随机种子
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# 创建回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)


# ==================== 1. 动态 λ 层 ====================
class GradientReversal(Layer):
    """Gradient Reversal Layer (GRL) 支持动态 λ"""
    def __init__(self, lambda_var, **kwargs):
        super().__init__(**kwargs)
        self.lambda_var = lambda_var          # 传入 tf.Variable

    @tf.custom_gradient
    def call(self, x):
        lambda_val = K.get_value(self.lambda_var)   # 取出当前值
        def grad(dy):
            return -lambda_val * dy
        return tf.identity(x), grad

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_var': self.lambda_var})
        return config


# ==================== 2. 条件域判别器（按类别） ====================

def build_cond_domain_branch(feat, logits, lambda_var):


    """feat:特征  logits:softmax前  lambda_var:tf.Variable
    """

    # 先 softmax 得到类别概率
    prob= tf.nn.softmax(logits)  # (B, C)
    prob= tf.expand_dims(prob, axis=1)  # (B, 1, C)
    feat_exp= tf.expand_dims(feat, axis=2)  # (B, D, 1)
    cond_feat= feat_exp * prob  # (B, D, C)  按类别加权
    cond_feat= tf.reduce_sum(cond_feat, axis=2)  # (B, D)     把类别维压缩

    grl= GradientReversal(lambda_var)
    x= grl(cond_feat)
    x= Dense(128, activation='relu')(x)
    x= Dropout(0.5)(x)
    x= Dense(64, activation='relu')(x)
    x= Dropout(0.5)(x)

    return Dense(1, activation='sigmoid')(x)


# 加载数据函数
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


def build_daan_model(input_shape, num_labels, lambda_dann=1.0):
    conv_layer1 = Conv1D(8, 80, strides=20, activation='relu')
    dropout_layer1 = Dropout(0.2)
    attention_layer1 = SelfAttention()
    flatten_layer1 = Flatten()
    dense_layer1 = Dense(128, activation='relu', name='feat1')
    classifier1 = Dense(num_labels, activation='softmax', name='cls1')

    conv_layer2 = Conv1D(8, 80, strides=20, activation='relu')
    dropout_layer2 = Dropout(0.2)
    attention_layer2 = SelfAttention()
    flatten_layer2 = Flatten()
    dense_layer2 = Dense(128, activation='relu', name='feat2')
    classifier2 = Dense(num_labels, activation='softmax', name='cls2')

    hp_lambda = tf.Variable(0.0, trainable=False, name='hp_lambda')

    def make_marg_domain_branch(feat):
        grl = GradientReversal(hp_lambda)
        x = grl(feat)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        return Dense(1, activation='sigmoid')(x)

    src_in = Input(shape=input_shape, name='src_input')
    tgt_in = Input(shape=input_shape, name='tgt_input')

    x_s = conv_layer1(src_in)
    x_s = dropout_layer1(x_s)
    x_s = attention_layer1(x_s)
    x_s = flatten_layer1(x_s)
    feat_s = dense_layer1(x_s)
    logits_s = classifier1(feat_s)
    dom_marg_s = make_marg_domain_branch(feat_s)

    x_t = conv_layer1(tgt_in)
    x_t = dropout_layer1(x_t)
    x_t = attention_layer1(x_t)
    x_t = flatten_layer1(x_t)
    feat_t = dense_layer1(x_t)
    logits_t = classifier2(feat_t)
    dom_marg_t = make_marg_domain_branch(feat_t)

    model = Model(inputs=[src_in, tgt_in],
                  outputs=[logits_s, logits_t, dom_marg_s, dom_marg_t])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',
                        'binary_crossentropy', 'binary_crossentropy'],
                  loss_weights=[1, 1, lambda_dann, lambda_dann],
                  optimizer=Adam(1e-4),
                  metrics=['accuracy'])
    model.summary()
    feature_ectractor = Model(inputs=[src_in, tgt_in], outputs=[feat_s, feat_t])
    return model, hp_lambda,feature_ectractor

def data_generator(X_s, y_s, X_t, y_t, batch_size=16):
    """生成器：同时提供源域和目标域数据及标签"""
    while True:
        ind_s = np.random.choice(len(X_s), 3*batch_size)
        ind_t = np.random.choice(len(X_t), batch_size)

        # 创建域标签（边缘 + 条件，各两组）
        batch_dom_s = np.ones((3*batch_size, 1))
        batch_dom_t = np.zeros((batch_size, 1))
        batch_dom_cond_s = np.ones((batch_size, 1))   # 条件分支标签
        batch_dom_cond_t = np.zeros((batch_size, 1))

        yield ([X_s[ind_s], X_t[ind_t]],
               [y_s[ind_s], y_t[ind_t],
                batch_dom_s, batch_dom_t])


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





def plot_pr_curve(y_true, y_scores, num_classes, title):
    # 计算每个类别的 Precision 和 Recall
    precision = dict()
    recall = dict()
    average_precision = dict()


    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_scores[:, i])


    # 绘制每个类别的 PR 曲线
    plt.figure(figsize=(10, 8))
    '''
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'Class {i} (AP = {average_precision[i]:.2f})')
    '''
    # 绘制微平均 PR 曲线
    y_true_bin = y_true.ravel()
    y_scores_bin = y_scores.ravel()
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin, y_scores_bin)
    print(precision["micro"])
    print(recall["micro"])
    average_precision["micro"] = average_precision_score(y_true_bin, y_scores_bin)
    plt.plot(recall["micro"], precision["micro"], color="gold", lw=2,
             label=f'Micro-average (AP = {average_precision["micro"]:.2f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt





def draw_tsne(X_src_test,X_tgt_test,y_src_test,y_tgt_test,title):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设 feat_src_test 和 feat_tgt_test 分别是源域测试集和目标域测试集的特征向量
    # 假设 y_src_test 和 y_tgt_test 分别是源域测试集和目标域测试集的标签
    # 这些特征向量和标签需要从你的模型中提取出来

    # 示例：假设 y_src_test 和 y_tgt_test 是标签数组
    # y_src_test = [0, 1, 2, 0, 1, 2, ...]
    # y_tgt_test = [0, 1, 2, 0, 1, 2, ...]

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
    X_src_test_2d = X_src_test.reshape(X_src_test.shape[0], -1)
    X_tgt_test_2d = X_tgt_test.reshape(X_tgt_test.shape[0], -1)

    # 合并展平后的特征
    all_feats = np.vstack((X_src_test_2d, X_tgt_test_2d))
    tsne = TSNE(n_components=2, random_state=42)
    all_feats_2d = tsne.fit_transform(all_feats)

    # 分离源域和目标域的特征
    src_feats_2d = all_feats_2d[:len(X_src_test_2d)]
    tgt_feats_2d = all_feats_2d[len(X_src_test_2d):]

    # 绘制 t-SNE 图
    plt.figure(figsize=(8, 8))

    # 绘制源域数据
    for label in np.unique(y_src_test):
        indices = np.where(y_src_test == label)[0]
        plt.scatter(src_feats_2d[indices, 0], src_feats_2d[indices, 1],
                    c=color_map[label], marker='o', label=f'Source Domain - Class {label}')

    # 绘制目标域数据
    for label in np.unique(y_tgt_test):
        indices = np.where(y_tgt_test == label)[0]
        plt.scatter(tgt_feats_2d[indices, 0], tgt_feats_2d[indices, 1],
                    c=color_map[label], marker='^', label=f'Target Domain - Class {label}')

    plt.title(title)
    #plt.legend()
    plt.show()




def draw_tsne2(X_src_test,X_tgt_test,y_src_test,y_tgt_test,title):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设 feat_src_test 和 feat_tgt_test 分别是源域测试集和目标域测试集的特征向量
    # 假设 y_src_test 和 y_tgt_test 分别是源域测试集和目标域测试集的标签
    # 这些特征向量和标签需要从你的模型中提取出来

    # 示例：假设 y_src_test 和 y_tgt_test 是标签数组
    # y_src_test = [0, 1, 2, 0, 1, 2, ...]
    # y_tgt_test = [0, 1, 2, 0, 1, 2, ...]

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
    #X_src_test_2d = X_src_test.reshape(X_src_test.shape[0], -1)
    #X_tgt_test_2d = X_tgt_test.reshape(X_tgt_test.shape[0], -1)
    X_src_test_2d = X_src_test
    X_tgt_test_2d = X_tgt_test

    # 合并展平后的特征
    all_feats = np.vstack((X_src_test_2d, X_tgt_test_2d))
    tsne = TSNE(n_components=2, random_state=42)
    all_feats_2d = tsne.fit_transform(all_feats)

    # 分离源域和目标域的特征
    src_feats_2d = all_feats_2d[:len(X_src_test_2d)]
    tgt_feats_2d = all_feats_2d[len(X_src_test_2d):]

    # 绘制 t-SNE 图
    plt.figure(figsize=(8, 8))

    # 绘制源域数据
    for label in np.unique(y_src_test):
        indices = np.where(y_src_test == label)[0]
        plt.scatter(src_feats_2d[indices, 0], src_feats_2d[indices, 1],
                    c=color_map[label], marker='o', label=f'Source Domain - Class {label}')

    # 绘制目标域数据
    for label in np.unique(y_tgt_test):
        indices = np.where(y_tgt_test == label)[0]
        plt.scatter(tgt_feats_2d[indices, 0], tgt_feats_2d[indices, 1],
                    c=color_map[label], marker='^', label=f'Target Domain - Class {label}')

    plt.title(title)
    # plt.legend()
    plt.show()




import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE


def draw_tsne3(X_src_test, X_tgt_test, y_src_test, y_tgt_test, title, output_file):
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
    X_src_test_2d = X_src_test.reshape(X_src_test.shape[0], -1)
    X_tgt_test_2d = X_tgt_test.reshape(X_tgt_test.shape[0], -1)

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

    src_val_file = 'E:\libs\hammer-newlabel4\\src_val\\src_val.csv'
    X_src_val, y_src_val_ = load_data(src_val_file)
    y_src_val = np.eye(17)[y_src_val_.astype(int)]

    # 加载目标域（训练）
    tgt_train_file = 'E:\libs\hammer-newlabel4\\transferlearningdata-new\\train1.0.csv'
    X_tgt_train, y_tgt_train_ = load_data(tgt_train_file)
    y_tgt_train = np.eye(17)[y_tgt_train_.astype(int)]
    tgt_val_file = 'E:\libs\hammer-newlabel4\\transferlearningdata/val0.1.csv'
    X_tgt_val, y_tgt_val_ = load_data(tgt_val_file)
    y_tgt_val = np.eye(17)[y_tgt_val_.astype(int)]
    tgt_test_file = 'E:\libs\hammer-newlabel4\\transferlearningdata-new\\0904+1119_new_test.csv'
    X_tgt_test, y_tgt_test_ = load_data(tgt_test_file)
    y_tgt_test = np.eye(17)[y_tgt_test_.astype(int)]

    # 划分目标域验证/测试集
    # X_tgt_val, X_tgt_test, y_tgt_val_, y_tgt_test_ = train_test_split(
    #     X_tgt_train, y_tgt_train_, test_size=0.8, random_state=seed_value)
    # y_tgt_val = np.eye(17)[y_tgt_val_.astype(int)]
    # y_tgt_test = np.eye(17)[y_tgt_test_.astype(int)]

    # 加载源域测试集
    src_test_file = 'E:\\libs\\hammer-newlabel4\\20250702-1.csv'
    X_src_test, y_src_test_ = load_data(src_test_file)
    y_src_test = np.eye(17)[y_src_test_.astype(int)]

    # 调整数据形状
    X_src = X_src.reshape(-1, 7062, 1)
    X_tgt_train = X_tgt_train.reshape(-1, 7062, 1)
    X_tgt_val = X_tgt_val.reshape(-1, 7062, 1)
    X_tgt_test = X_tgt_test.reshape(-1, 7062, 1)
    X_src_test = X_src_test.reshape(-1, 7062, 1)

    # 目标函数
    def objective(trial):
        lambda_dann = trial.suggest_float('lambda_dann', 0.001, 10.0, log=True)
        model, hp_lambda, _ = build_daan_model((7062, 1), 17, lambda_dann=lambda_dann)

        #for v in model.trainable_variables:
        #    print(v.name)
        steps = max(len(X_src), len(X_tgt_train)) // 16

        # 动态 λ 更新函数
        def scheduler(epoch):
            p = epoch / 20  # 总 epoch 20 为例
            new_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            K.set_value(hp_lambda, new_lambda)

        lambda_cb = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: scheduler(epoch))

        # 验证数据域标签
        val_dom_s = np.ones((len(y_tgt_train), 1))
        val_dom_t = np.zeros((len(y_tgt_train), 1))
        #val_dom_s[:] = 0.5
        #val_dom_t[:] = 0.5

        model.fit(
            data_generator(X_src, y_src, X_tgt_train, y_tgt_train, 16),
            steps_per_epoch=steps,
            epochs=20,
            validation_data=([X_src_val[:len(X_tgt_train)], X_tgt_train],
                             [y_src_val[:len(y_tgt_train)], y_tgt_train,
                              val_dom_s, val_dom_t]),
            verbose=0,
            callbacks=[lambda_cb])

        # 评估
        dummy_src = np.zeros_like(X_tgt_train)
        eval_results = model.evaluate([dummy_src, X_tgt_train],
                                      [y_src_val[:len(y_tgt_train)], y_tgt_train,
                                       val_dom_s, val_dom_t],
                                      verbose=0)
        print(model.metrics_names)
        print(eval_results)


        return eval_results[5]+eval_results[6]


    # Optuna调参
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=1, n_jobs=1)

    best_lambda = study.best_params['lambda_dann']
    print("Best lambda_dann:", best_lambda)

    # 最终训练
    best_model, _ ,feature_extractor= build_daan_model((7062, 1), 17, lambda_dann=best_lambda)
    steps = max(len(X_src), len(X_tgt_train)) // 16

    # ✅ 在这里定义缺失的变量
    val_dom_s = np.ones((len(y_tgt_train), 1))
    val_dom_t = np.zeros((len(y_tgt_train), 1))
    #val_dom_s[:] = 0.5
    #val_dom_t[:] = 0.5

    best_model.fit(
        data_generator(X_src, y_src, X_tgt_train, y_tgt_train, 16),
        steps_per_epoch=steps,
        epochs=20,
        validation_data=([X_src_val[:len(X_tgt_train)], X_tgt_train],
                         [y_src_val[:len(y_tgt_train)], y_tgt_train,
                          val_dom_s, val_dom_t]),
        callbacks=[reduce_lr])


    # 目标域评估
    dummy_src = np.zeros_like(X_tgt_test)
    y_pred_tgt = best_model.predict([dummy_src, X_tgt_test])[1]
    y_pred_tgt_class = np.argmax(y_pred_tgt, axis=1)
    y_tgt_test_class = np.argmax(y_tgt_test, axis=1)

    acc_tgt = accuracy_score(y_tgt_test_class, y_pred_tgt_class)
    weighted_f1_tgt = calculate_weighted_f1(y_tgt_test_class, y_pred_tgt_class, num_classes=17)
    micro_f1_tgt = calculate_micro_f1(y_tgt_test_class, y_pred_tgt_class, num_classes=17)
    macro_f1_tgt = calculate_macro_f1(y_tgt_test_class, y_pred_tgt_class, num_classes=17)
    print(f"Target domain accuracy: {acc_tgt:.4f}")
    print(f"Target domain weighted F1 Score: {weighted_f1_tgt:.4f}")
    print(f"Target domain micro F1 Score: {micro_f1_tgt:.4f}")
    print(f"Target domain macro F1 Score: {macro_f1_tgt:.4f}")

    # 绘制目标域 PR 曲线
    #plot_pr_curve(y_tgt_test, y_pred_tgt, num_classes=17, title="Precision-Recall Curve - Target Domain")



    # 源域评估
    y_pred_src = best_model.predict([X_src_test, X_src_test])[0]
    y_pred_src_class = np.argmax(y_pred_src, axis=1)
    y_src_test_class = np.argmax(y_src_test, axis=1)

    acc_src = accuracy_score(y_src_test_class, y_pred_src_class)
    weighted_f1_src = calculate_weighted_f1(y_src_test_class, y_pred_src_class, num_classes=17)
    micro_f1_src = calculate_micro_f1(y_src_test_class, y_pred_src_class, num_classes=17)
    macro_f1_src = calculate_macro_f1(y_src_test_class, y_pred_src_class, num_classes=17)
    print(f"Source domain accuracy: {acc_src:.4f}")
    print(f"Source domain weighted F1 Score: {weighted_f1_src:.4f}")
    print(f"Source domain micro F1 Score: {micro_f1_src:.4f}")
    print(f"Source domain macro F1 Score: {macro_f1_src:.4f}")

    # 绘制源域 PR 曲线
    #plot_pr_curve(y_src_test, y_pred_src, num_classes=17, title="Precision-Recall Curve - Source Domain")

    '''
    # 混淆矩阵
    conf_matrix = confusion_matrix(np.argmax(y_tgt_test, 1), np.argmax(y_pred_tgt, 1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Target Domain')
    #plt.show()
    '''

    #draw_tsne(X_src_test, X_tgt_test, y_src_test_, y_tgt_test_,"Original Data")

    feat_src_test = feature_extractor([X_src_test, X_src_test])[0]
    feat_tgt_test = feature_extractor([X_tgt_test, X_tgt_test])[1]
    draw_tsne4(feat_src_test, feat_tgt_test, y_src_test_, y_tgt_test_,"Embedding","DDAN.html")






if __name__ == '__main__':
    starttime = time.time()
    main()
    endtime = time.time()
    print(endtime-starttime)

    winsound.Beep(2500, 1000)