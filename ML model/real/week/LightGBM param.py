import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from alltrain import split_data

# 创建LightGBM分类器对象
lgb_clf = lgb.LGBMClassifier(random_state=42,n_jobs=10)
# 定义标签值
labels = [0, 1, 2, 3, 4]
# 将标签0视为正类，将标签1-4视为负类
positive_class = 0
negative_classes = [1, 2, 3, 4]
# 定义需要移除的特征列
removed_cols = ['user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider']
# 从csv文件中读取数据
filename = 'day-r5.2.csv.gz'
data = pd.read_csv(filename)
print('data load successfully')

# 定义要搜索的参数组合
param_dist = {
    'num_leaves': sp_randint(20, 200),
    'learning_rate': [0.1],
    'n_estimators': sp_randint(50, 500),
    'min_child_samples': sp_randint(10, 30),
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [0.1, 0.5]
}
res = split_data(data, test_size=0.25, random_state=0, y_column='insider',
                 shuffle=True,
                 x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid',
                            'timeind', 'Unnamed: 0', 'insider'),
                 dname='r5.2', normalization='StandardScaler',
                 rm_empty_cols=True, by_user=True, by_user_time=True,
                 by_user_time_trainper=0.5, limit_ntrain_user=0)

# print(res)
# 从预处理后的数据集中获取训练集和测试集数据
x_train = res['x_train']
y_train = res['y_train']
x_test = res['x_test']
y_test = res['y_test']
# 创建RandomizedSearchCV对象，指定要搜索的参数范围和评分标准
print('start')
random_search = RandomizedSearchCV(lgb_clf, param_distributions=param_dist, n_iter=50,
                                   scoring="accuracy", cv=5, n_jobs=10, random_state=42)


# 在训练集上进行随机搜索
random_search.fit(x_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found: ", random_search.best_params_)
print("Best ROC AUC score found: ", random_search.best_score_)

# 使用最佳参数训练模型并预测测试集
lgb_clf = lgb.LGBMClassifier(random_state=42, **random_search.best_params_)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
y_prob = lgb_clf.predict_proba(x_test)[:, 1]

# 构建二进制标签
y_test_binary = [1 if label == positive_class else 0 for label in y_test]
y_pred_binary = [1 if label == positive_class else 0 for label in y_pred]
y_prob_binary = [prob[positive_class] for prob in y_prob]
# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob_binary)
auc_score = auc(fpr, tpr)
print('ROC AUC Score:', auc_score)

# 绘制ROC曲线图像
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of LightGBM classifier')
plt.legend(loc="lower right")
plt.savefig('LightGBM classifier roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()