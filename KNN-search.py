from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import random as python_random
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import average_precision_score
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 添加参数搜索
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform  # 用于RandomizedSearchCV的分布

starttime = time.time()


# ========== 原有函数保持不变 ==========

def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    feature_names = combined_data.columns[:-1].tolist()
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1:].values
    return X, y, feature_names


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1:].values
    return X, y


def normalize_samples(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        X_normalized[i, :] = (X[i, :] - mean) / std
    return X_normalized


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
    n_classes = y_true.shape[1]
    aps = []
    for c in range(n_classes):
        ap = average_precision_score(y_true[:, c], y_pred[:, c])
        if not np.isnan(ap):
            aps.append(ap)
    return aps


def calculate_map(aps):
    return np.mean(aps) if aps else 0.0


def minmax_per_sample(X):
    X_scaled = np.zeros_like(X, dtype=float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(X.shape[0]):
        X_scaled[i:i + 1, :] = scaler.fit_transform(X[i:i + 1, :])
    return X_scaled


def load_selected_features(file_path):
    selected_features_df = pd.read_excel(file_path, header=None)
    selected_features = selected_features_df.iloc[:, 0].tolist()
    return selected_features


def filter_features(X, feature_names, selected_features):
    feature_names_float = [float(name) for name in feature_names]
    selected_features_float = [float(feature) for feature in selected_features]
    selected_indices = [feature_names_float.index(feature) for feature in selected_features_float if
                        feature in feature_names_float]
    X_filtered = X[:, selected_indices]
    return X_filtered, selected_indices


def calculate_macro_metrics(y_true, y_pred):
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return macro_precision, macro_recall, macro_f1


# ========== 新增：参数搜索函数 ==========

def knn_grid_search(X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1):
    """
    使用GridSearchCV搜索KNN最优参数（网格搜索）

    参数:
        cv: 交叉验证折数
        scoring: 评分标准 ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'等)
        n_jobs: 并行作业数，-1表示使用所有CPU核心
    """
    # 定义参数网格
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25, 31, 45, 51, 61, 71, 81, 91, 101],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean'],
        'p': [1, 2, 3]  # Minkowski距离的幂参数
    }

    # 创建KNN分类器
    knn = KNeighborsClassifier()

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,  # 输出搜索进度
        return_train_score=True
    )

    print("开始Grid Search参数搜索...")
    print(f"参数空间大小: {np.prod([len(v) for v in param_grid.values()])} 种组合")
    print(f"交叉验证折数: {cv}")

    # 执行搜索（注意：y_train需要flatten）
    grid_search.fit(X_train, y_train.ravel())

    print("\n========== Grid Search 结果 ==========")
    print(f"最优参数: {grid_search.best_params_}")
    print(f"最优交叉验证得分: {grid_search.best_score_:.4f}")

    # 输出前5个最佳结果
    results_df = pd.DataFrame(grid_search.cv_results_)
    print("\n前5个最佳参数组合:")
    top5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    for idx, row in top5.iterrows():
        print(f"  参数: {row['params']}, 得分: {row['mean_test_score']:.4f} (+/- {row['std_test_score'] * 2:.4f})")

    return grid_search.best_estimator_, grid_search.best_params_


def knn_random_search(X_train, y_train, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=44):
    """
    使用RandomizedSearchCV搜索KNN最优参数（随机搜索，适合大参数空间）

    参数:
        n_iter: 随机采样参数组合的数量
        cv: 交叉验证折数
        scoring: 评分标准
        n_jobs: 并行作业数
        random_state: 随机种子
    """
    # 定义参数分布
    param_distributions = {
        'n_neighbors': randint(3, 101),  # 3到100之间的随机整数
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'p': randint(1, 5)  # Minkowski距离的幂参数
    }

    # 创建KNN分类器
    knn = KNeighborsClassifier()

    # 创建RandomizedSearchCV对象
    random_search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_distributions,
        n_iter=n_iter,  # 随机采样n_iter个组合
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        random_state=random_state,
        return_train_score=True
    )

    print("开始Random Search参数搜索...")
    print(f"随机采样数: {n_iter}")
    print(f"交叉验证折数: {cv}")

    # 执行搜索
    random_search.fit(X_train, y_train.ravel())

    print("\n========== Random Search 结果 ==========")
    print(f"最优参数: {random_search.best_params_}")
    print(f"最优交叉验证得分: {random_search.best_score_:.4f}")

    # 输出前5个最佳结果
    results_df = pd.DataFrame(random_search.cv_results_)
    print("\n前5个最佳参数组合:")
    top5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    for idx, row in top5.iterrows():
        print(f"  参数: {row['params']}, 得分: {row['mean_test_score']:.4f} (+/- {row['std_test_score'] * 2:.4f})")

    return random_search.best_estimator_, random_search.best_params_


# ========== 主程序 ==========

if __name__ == "__main__":
    # 设置随机数种子
    seed_value = 44
    set_random_seeds(seed_value)

    # 指定训练集和测试集文件
    train_csv = [
        'E:\libs\hammer-newlabel4\\20250311.csv',
        'E:\libs\hammer-newlabel4\\0626-2.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        'E:\libs\hammer-newlabel4\\20250704-2.csv'
    ]
    test_csv = 'E:\libs\hammer-newlabel4\\20250702-1.csv'

    # 加载训练数据
    X_train, y_train, feature_names = load_data_from_files(train_csv)

    # 加载测试数据
    X_test, y_test = load_data(test_csv)

    guiyihua = "False"#normalization

    if guiyihua == "False":
        pass
    else:
        X_train = minmax_per_sample(X_train)
        X_test = minmax_per_sample(X_test)

    # ========== 参数搜索部分（二选一） ==========

    # 方式1: 网格搜索 - 适合参数空间较小的情况
    best_knn, best_params = knn_grid_search(
         X_train, y_train,
         cv=5,
         scoring='f1_macro',  # 可选: 'f1_macro', 'precision_macro', 'recall_macro'
         n_jobs=1#-1是多线程，1是单线程
    )

    # 方式2: 随机搜索 - 适合参数空间较大的情况，速度更快
    #best_knn, best_params = knn_random_search(
    #    X_train, y_train,
    #    n_iter=30,  # 随机尝试30种组合
    #    cv=5,
    #    scoring='accuracy',
    #    n_jobs=1,#-1是多线程，1是单线程
    #    random_state=seed_value
    #)

    # 使用最优模型进行预测
    print("\n========== 在测试集上评估最优模型 ==========")
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)

    # 评估
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))
    print("KNN Classification Report:\n", classification_report(y_test, y_pred))

    # 计算宏平均精确率、召回率和F1值
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(y_test, y_pred)
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
    plt.title('Confusion Matrix-KNN-Optimized')
    plt.show()

    endtime = time.time()
    print("Total time taken:", endtime - starttime)