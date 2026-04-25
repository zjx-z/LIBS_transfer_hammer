


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
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
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


# 设置随机数种子
seed_value = 44
set_random_seeds(seed_value)

# 指定训练集和测试集文件
train_csv = [
    # 'E:\libs\hammer - newlabel3\\20250311.csv',
    # 'E:\libs\hammer - newlabel3\\0626-2.csv',
    # 'E:\libs\hammer - newlabel3\\20250702-1.csv',
    # 'E:\libs\hammer - newlabel3\\20250702-2.csv',
    # 'E:\libs\hammer - newlabel3\\20250703.csv',
    # 'E:\libs\hammer - newlabel3\\20250704-1.csv',
    # 'E:\libs\hammer - newlabel3\\20250704-2.csv',
    #'E:\\libs\\20250702_1+20250702_2+20250703+20250704_1(no safehammer).csv'
        'E:\libs\hammer-newlabel4\\20250311.csv',
        'E:\libs\hammer-newlabel4\\0626-2.csv',
        #'E:\libs\hammer-newlabel4\\20250702-1.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        'E:\libs\hammer-newlabel4\\20250704-2.csv',
]  # 训练 + 验证
test_csv = 'E:\libs\hammer-newlabel4\\20250702-1.csv'  # 测试

# 加载训练数据
X_train, y_train, feature_names = load_data_from_files(train_csv)

# 加载测试数据
X_test, y_test= load_data(test_csv)

guiyihua = "False"

if guiyihua == "False":
    pass
else:
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    #X_scaled=normalize_samples(X_train)
    # 逐样本 0-1 归一化
    X_train = minmax_per_sample(X_train)
    X_test = minmax_per_sample(X_test)





# # 加载指定特征名称
# selected_features_file = 'E:\libs\code\mycode\experiments\\newlabel_manselected_features.xlsx'  # 替换为你的Excel文件路径
# selected_features = load_selected_features(selected_features_file)
#
# # 筛选指定特征
# X_train_filtered, selected_indices = filter_features(X_train, feature_names, selected_features)
# X_test_filtered, _ = filter_features(X_test, feature_names, selected_features)
#
# # 转换为 2D
# X_train_2d = X_train_filtered.reshape(X_train_filtered.shape[0], -1)
# X_test_2d = X_test_filtered.reshape(X_test_filtered.shape[0], -1)



from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 定义参数网格（针对LIBS光谱数据优化）
param_grid = {
    'C': [0.1, 1, 10],           # 正则化参数，控制模型复杂度
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # 核函数系数
    'kernel': ['rbf']              # 核函数类型（rbf适合高维光谱数据）
}

print("开始SVM参数搜索（Grid Search）...")
# 使用宏平均F1作为评估指标，适合多分类问题
grid_search = GridSearchCV(
    SVC(probability=True, random_state=seed_value),
    param_grid,
    scoring='f1_macro',    # 或 'accuracy', 'precision_macro', 'recall_macro'
    cv=5,                  # 5折交叉验证（样本少可改为3折）
    n_jobs=1,             # 使用所有CPU并行计算
    verbose=2              # 显示详细进度（每个参数组合都会输出）
)

grid_search.fit(X_train, y_train.ravel())  # y_train需要展平为一维

# 获取最佳模型
svm = grid_search.best_estimator_

print(f"\n最佳参数组合：{grid_search.best_params_}")
print(f"最佳交叉验证F1分数：{grid_search.best_score_:.4f}")



from joblib import dump, load
#dump(svm, "svm.pkl")



# 预测
# y_pred = rf.predict(X_test_2d)
# y_prob = rf.predict_proba(X_test_2d)
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)

# 评估
#print("SVM Accuracy:", accuracy_score(y_test, y_pred))

# 计算宏平均精确率、召回率和F1值
macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test, y_pred)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print(f"Macro-average Precision: {macro_precision:.4f}")
print(f"Macro-average Recall: {macro_recall:.4f}")
print(f"Macro-average F1 Score: {macro_f1:.4f}")


# print("SVM Classification Report:\n", classification_report(y_test, y_pred))
#
# for _ in range(len(y_test)):
#     print(y_test[_][0],y_pred[_])
# # Rank 指标
# rank_metrics = calculate_rank_metrics(y_test, y_prob)
# for rank, value in rank_metrics.items():
#     print(f"{rank}: {value:.4f}")

# # 计算 Rank 指标
# rank_metrics = calculate_rank_metrics(y_test, y_pred)
# for rank, value in rank_metrics.items():
#     #print(f"{rank}: {value:.4f}")
#     print(f"{value:.4f}")
#     pass
'''
# 计算 per-class AP
aps = calculate_per_class_ap(y_test, y_prob)
mAP = calculate_map(aps)

# 打印每个类别的 AP 和 mAP
for c, ap in enumerate(aps):
    #print(f"Class {c + 1} AP: {ap:.4f}")
    pass
print(f"mAP: {mAP:.4f}")
'''
'''
importances=rf.feature_importances_
feat_imp=pd.Series(importances,index=feature_names).sort_values(ascending=False)

# print("特征重要性排序：")
# print(feat_imp)
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
feat_imp_df.to_excel('feature_importance_Al_newlabel0704-1.xlsx', index=False)
print("特征重要性已保存为 'feature_importance_Al_newlabel0704-1.xlsx'")
'''
'''
import seaborn as sns
import matplotlib.pyplot as plt
#
# sns.barplot(x=feat_imp[:30], y=feat_imp.index[:30])
# plt.title("Feature Importance from Random Forest")
# plt.show()

wavelengths = np.array([float(name) for name in feature_names])  # 如果特征名是波长
#fig, ax1 = plt.subplots(figsize=(12, 5))

# 左轴：光谱强度（线图）
# ax1.plot(wavelengths, importances, color='tab:blue')
# ax1.set_xlabel("Wavelength (nm)")
# ax1.set_ylabel("feature importance", color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# fig.tight_layout()
# plt.show()

X,y=load_data("E:\libs\spectra_samples.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 上图：所有样本的光谱曲线
for spectrum in X:
    ax1.plot(wavelengths, spectrum, linewidth=0.6, alpha=0.7)
ax1.set_ylabel('Intensity')
ax1.set_title('All sample spectra')

# 下图：RF 特征重要性
ax2.bar(wavelengths, importances,
        width=np.diff(wavelengths).mean()*0.8,
        color='firebrick', alpha=0.8)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('RF importance')
ax2.set_title('Feature importance from Random Forest-traindata3456no_safehammer')

plt.tight_layout()
plt.show()
plt.savefig('plot.png')
'''
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)
plt.rcParams['font.family'] = 'Times New Roman'
# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted Labels', labelpad=40)
plt.ylabel('True Labels', labelpad=20)
plt.title('Confusion Matrix-SVM-Dataset7')
#plt.savefig('Confusion Matrix-SVM-Dataset7.png', dpi=1200)  # 保存图像，设置分辨率为 600 dpi
#plt.show()


endtime=time.time()
print(endtime-starttime)

# 程序运行结束，发出声音
import winsound
winsound.Beep(2500, 1000)  # 频率 2500 Hz，持续时间 1000 毫秒