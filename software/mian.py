# 'week', 'day', 'session', 'subsession-nact25', 'subsession-nact50', 'subsession-time120', 'subsession-time240'
import sys
import os
import time

import pandas as pd
import numpy as np
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, \
    QComboBox, QDesktopWidget, QScrollArea, QTextEdit
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler




def split_data(data, y_column='insider', x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid',
                                                    'timeind', 'Unnamed: 0', 'insider'), rm_empty_cols=True):
    """
    split data into train and test sets or use all data for training, with normalization builtin
    """

    x_cols = [i for i in data.columns if i not in x_rm_cols]
    if rm_empty_cols:
        x_cols = [i for i in x_cols if len(set(data[i])) > 1]

    infocols = list(set(data.columns) - set(x_cols))

    # normalization
    sc = StandardScaler()


    # Use all data for training
    x_train = data[x_cols].values
    y_train = data[y_column].values

    x_train = sc.fit_transform(x_train)


    return {'x': x_train, 'y': y_train, 'sc': sc, 'infocols': infocols}

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.x_test=[]
        self.y_test=[]
        self.y_pred=[]
        self.file_type = None
        self.clf = None
        self.models = {
            'week': 'LightGBM.pkl',
            'day': 'LightGBM.pkl',
            'session': 'LightGBM.pkl',
            'subsession-nact25': 'LightGBM.pkl',
            'subsession-nact50': 'LightGBM.pkl',
            'subsession-time120': 'LightGBM.pkl',
            'subsession-time240': 'LightGBM.pkl'
        }

        self.initUI()

    def initUI(self):
        # 设置窗口属性
        self.setWindowTitle('机器学习预测程序')
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(1200, 800)
        self.center()

        # 创建QTextEdit部件
        self.text_edit = QTextEdit()


        # 创建滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)  # 设置滚动区域自适应大小
        self.setLayout(QVBoxLayout(self))  # 创建主窗口布局
        self.layout().addWidget(scroll_area)  # 将滚动区域添加到主窗口布局中
        scroll_area.setWidget(self.text_edit)

        # 创建一个widget来存放所有控件
        widget = QWidget(scroll_area)
        scroll_area.setWidget(widget)

        # 添加控件
        self.file_label = QLabel('请选择数据文件', widget)
        self.file_label.setAlignment(Qt.AlignCenter)

        self.open_file_button = QPushButton('打开文件', widget)
        self.open_file_button.clicked.connect(self.open_file)

        self.type_label = QLabel('请选择数据类型', widget)
        self.type_label.setAlignment(Qt.AlignCenter)

        self.type_combo_box = QComboBox(widget)
        self.type_combo_box.addItems(
            ['week', 'day', 'session', 'subsession-nact25', 'subsession-nact50', 'subsession-time120',
             'subsession-time240'])  # 添加文件类型选项
        self.type_combo_box.currentIndexChanged.connect(self.update_model)

        self.predict_button = QPushButton('预测', widget)
        self.predict_button.clicked.connect(self.predict)

        self.text_edit = QTextEdit('请先选择数据文件和数据类型，并加载相应的机器学习模型', widget)
        self.text_edit.setAlignment(Qt.AlignCenter)
        self.text_edit.setFont(QFont("Arial", 12))


        # 添加布局
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.file_label)
        hbox1.addWidget(self.open_file_button)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.type_label)
        hbox2.addWidget(self.type_combo_box)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.predict_button)

        vbox = QVBoxLayout(widget)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addWidget(self.text_edit)

        # 将widget设置为滚动区域的子控件
        widget.setLayout(vbox)

    def center(self):
        # 让窗口居中显示
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def open_file(self):
        # 打开数据文件对话框，选择数据文件
        file_path, _ = QFileDialog.getOpenFileName(self, '打开数据文件', os.getcwd())
        print(file_path)
        if file_path:
            self.file_label.setText('已选择文件：{}'.format(file_path))

            # 读取数据文件
            try:
                data = pd.read_csv(file_path)
                self.text_edit.setPlainText('已成功读取数据文件')
                res = split_data(data,  y_column='insider', x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid','timeind', 'Unnamed: 0', 'insider'),rm_empty_cols=True)
                self.x_test = res['x']
                print("x_test ", len(self.x_test))
                self.y_test = res['y']

            except Exception as e:
                print(e)
                self.text_edit.setPlainText('发生错误：{}'.format(e))
                return


    def update_model(self):
        # 根据选中的文件类型更新机器学习模型
        file_type = self.type_combo_box.currentText()
        print(file_type)
        if file_type in self.models:
            model_file = self.models[file_type]
            try:
                with open(model_file, 'rb') as f:
                    self.clf = joblib.load(f)
                print(self.clf)
                self.file_type = file_type
                self.text_edit.setPlainText('已成功更新机器学习模型：{}'.format(model_file))
            except Exception as e:
                print(e)
                self.text_edit.setPlainText('发生错误：{}'.format(e))

    def predict(self):
        print(self.y_test)
        # 预测数据并输出结果
        if self.clf is None:
            self.text_edit.setPlainText('请先选择数据文件和数据类型，并加载相应的机器学习模型')
            return
        n = self.clf.feature_importances_.shape[0]

        rows, cols = self.x_test.shape

        # 如果列数不够，用0填充
        if cols < n:
            self.x_test = np.pad(self.x_test, ((0, 0), (0, n - cols)), 'constant')
        # 如果列数超过，删除多余的列
        elif cols > n:
            self.x_test = self.x_test[:, :n]

        # 进行预测
        self.y_pred = self.clf.predict(self.x_test)
        print(self.y_pred)
        y_prob = self.clf.predict_proba(self.x_test)
        # 输出分类报告
        print('Classification Report')
        print(classification_report(self.y_test, self.y_pred))
        # 构建二进制标签
        y_test_binary = [1 if label == 0 else 0 for label in self.y_test]
        y_pred_binary = [1 if label == 0 else 0 for label in self.y_pred]

        # 计算混淆矩阵及其性能指标
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        print('Confusion Matrix')
        print(cm)

        # 获取TN，FP，FN和TP的值
        tn, fp, fn, tp = cm.ravel()

        fpr1 = fp / (fp + tn)
        pr = tp / (tp + fp)
        f1 = f1_score(y_test_binary, y_pred_binary)

        print('False Positive Rate:', fpr1)
        print('Precision:', pr)
        print('F1 Score:', f1)

        # 计算ROC曲线和AUC值
        y_prob_binary = [prob[0] for prob in y_prob]
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
        plt.title('ROC curve of ' + self.file_type )
        plt.legend(loc="lower right")
        plt.show()
        # plt.savefig(self.file_type  + '-' + 'roc_curve.png', dpi=300, bbox_inches='tight')


        self.text_edit.setPlainText('Classification Report\n{}\nConfusion Matrix\n{}\n\nFalse Positive Rate:{}\nPrecision:{}\nF1 Score:{}\nROC AUC Score:{}\n预测结果：{}'.format(classification_report(self.y_test,self.y_pred),cm,str(fpr1),str(pr),str(f1),str(auc_score),self.y_pred))



    def keyPressEvent(self, event):
        # 放大页面时，按键也要同步
        if event.key() == Qt.Key_Control:
            self.resize(500, 400)

    def keyReleaseEvent(self, event):
        # 放大页面时，按键也要同步
        if event.key() == Qt.Key_Control:
            self.resize(400, 300)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())