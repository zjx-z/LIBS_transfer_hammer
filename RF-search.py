




from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow.keras.backend as K
import random as python_random
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

starttime=time.time()
# 加载多个数据文件
def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    feature_names = combined_data.columns[:-1].tolist()  # 除了最后一列的特征名
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1:].values
    return X, y ,feature_names

# 加载数据
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1:].values
    return X, y


# 自定义对每个样本进行归一化
def normalize_samples(X):
    """
    对每个样本进行归一化，使得每个样本的均值为 0，标准差为 1。
    """
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        X_normalized[i, :] = (X[i, :] - mean) / std
    return X_normalized


# 计算 Rank 指标
def calculate_rank_metrics(y_true, y_pred):
    ranks = [1, 3, 5, 10, 20]
    results = {}
    for rank in ranks:
        correct = 0
        for i in range(len(y_true)):
            top_k_indices = np.argsort(y_pred[i])[-rank:][::-1]
            if np.argmax(y_true[i]) in top_k_indices:
                correct += 1
        results[f'Rank-{rank}'] = correct / len(y_true)
    return results

def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)



def calculate_per_class_ap(y_true, y_pred):
    """
    计算每个类别的 Average Precision (AP)
    :param y_true: 真实标签 (one-hot 编码)
    :param y_pred: 预测概率
    :return: 每个类别的 AP 列表
    """
    n_classes = y_true.shape[1]
    aps = []
    for c in range(n_classes):
        ap = average_precision_score(y_true[:, c], y_pred[:, c])
        if not np.isnan(ap):  # 如果某类没有正样本，AP 会是 NaN，跳过
            aps.append(ap)
    return aps

def calculate_map(aps):
    """
    计算 mAP
    :param aps: 每个类别的 AP 列表
    :return: mAP
    """
    return np.mean(aps) if aps else 0.0

from sklearn.preprocessing import MinMaxScaler

def minmax_per_sample(X):
    """
    把每一行（每个样本）线性归一化到 [0,1]
    """
    X_scaled = np.zeros_like(X, dtype=float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(X.shape[0]):
        X_scaled[i:i+1, :] = scaler.fit_transform(X[i:i+1, :])
    return X_scaled


# 加载指定特征名称
def load_selected_features(file_path):
    selected_features_df = pd.read_excel(file_path, header=None)
    selected_features = selected_features_df.iloc[:, 0].tolist()
    return selected_features

# 筛选指定特征
def filter_features(X, feature_names, selected_features):
    """
    筛选指定特征。
    :param X: 原始数据矩阵 (numpy array)
    :param feature_names: 原始特征名称列表 (list of str)
    :param selected_features: 指定的特征名称列表 (list of str)
    :return: 筛选后的数据矩阵 (numpy array)，筛选后的特征索引 (list of int)
    """
    # 将特征名称转换为浮点数，以便进行精确匹配
    feature_names_float = [float(name) for name in feature_names]
    selected_features_float = [float(feature) for feature in selected_features]

    # 找到指定特征在原始特征列表中的索引
    selected_indices = [feature_names_float.index(feature) for feature in selected_features_float if feature in feature_names_float]

    # 根据索引筛选特征
    X_filtered = X[:, selected_indices]

    return X_filtered, selected_indices

# 计算宏平均精确率、宏平均召回率和宏平均F1值
def calculate_macro_metrics(y_true, y_pred):
    """
    计算宏平均精确率、召回率和F1分数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 宏平均精确率、召回率和F1分数
    """
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return macro_precision, macro_recall, macro_f1

# 在主函数中调用这些函数并输出结果
if __name__ == "__main__":
    # 设置随机数种子
    seed_value = 44
    set_random_seeds(seed_value)

    # 指定训练集和测试集文件
    train_csv = [
        'E:\libs\hammer-newlabel4\\20250311.csv',
        'E:\libs\hammer-newlabel4\\0626-2.csv',
        #'E:\libs\hammer-newlabel4\\20250702-1.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        'E:\libs\hammer-newlabel4\\20250704-2.csv',
        #'E:\libs\hammer-newlabel4\\20250904-train+val.csv'
    ]  # 训练 + 验证
    test_csv = 'E:\libs\hammer-newlabel4\\20250702-1.csv'  # 测试
    #test_csv='E:\libs\hammer-newlabel4\\20250904-test.csv'

    # 加载训练数据
    X_train, y_train, feature_names = load_data_from_files(train_csv)

    # 加载测试数据
    X_test, y_test = load_data(test_csv)

    guiyihua = "False"

    if guiyihua == "False":
        pass
    else:
        X_train = minmax_per_sample(X_train)
        X_test = minmax_per_sample(X_test)

    # 参数搜索训练（替换原来的rf训练）
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    print("开始参数搜索（针对LIBS光谱数据优化）...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=seed_value, n_jobs=1),
        param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=1,
        verbose=2
    )

    # 注意：y_train需要展平（ravel），因为RandomForest需要一维标签
    grid_search.fit(X_train, y_train.ravel())
    rf = grid_search.best_estimator_

    print(f"\n最佳参数：{grid_search.best_params_}")
    print(f"最佳CV分数：{grid_search.best_score_:.4f}")



    # 预测
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    # 评估
    print("RF Accuracy:", accuracy_score(y_test, y_pred))
    print("RF Classification Report:\n", classification_report(y_test, y_pred))

    # 计算宏平均精确率、召回率和F1值
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test, y_pred)
    print("RF Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.xlabel('Predicted Labels', labelpad=40)
    plt.ylabel('True Labels', labelpad=20)
    plt.title('Confusion Matrix-RF-TrLIBS')
    plt.show()

    endtime = time.time()
    print("Total time taken:", endtime - starttime)