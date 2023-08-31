import os
import json
import argparse
import sys
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prettytable import PrettyTable
import my_transforms
from utils import read_split_data_seq, cross_validation
from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from model import MLP
from sklearn.metrics import roc_curve, auc
from itertools import cycle


class CombinedModel(torch.nn.Module):
    def __init__(self, model, mlp):
        super(CombinedModel, self).__init__()
        self.model = model
        self.mlp = mlp

    def forward(self, x):
        _, x = self.model(x)
        return self.mlp(x)


def plot_multiclass_roc(all_labels, all_scores, num_classes):
    """
    Plot ROC curves for multiclass classification.

    Args:
        all_labels: ground truth labels.
        all_scores: predicted scores for each class.
        num_classes: total number of classes.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    all_labels_onehot = np.eye(num_classes)[all_labels]

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_onehot[:, i], np.array(all_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def compute_macro_avg_roc_auc(all_labels, all_scores, num_classes):
    """
    Compute ROC and AUC for multiclass classification.

    Args:
        all_labels: ground truth labels.
        all_scores: predicted scores for each class.
        num_classes: total number of classes.

    Returns:
        fpr: false positive rates for each class.
        tpr: true positive rates for each class.
        roc_auc: AUC values for each class.
        macro_auc: macro-average AUC value.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    all_labels_onehot = np.eye(num_classes)[all_labels]

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_onehot[:, i], np.array(all_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes

    macro_auc = auc(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, macro_auc


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1 = round((2 * Precision * Recall) / (Precision + Recall),3) if Precision + Recall != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1, Specificity])
        print(table)

    def overall_summary_micro(self):
        # total TP, FP, and FN
        TP = np.trace(self.matrix)
        FP = np.sum(np.sum(self.matrix, axis=0) - np.diagonal(self.matrix))
        FN = np.sum(np.sum(self.matrix, axis=1) - np.diagonal(self.matrix))
        TN = np.sum(self.matrix) - TP - FP - FN

        # calculate accuracy
        acc = TP / np.sum(self.matrix)
        # precision, recall and F1 score
        Precision = TP / (TP + FP) if TP + FP != 0 else 0.
        Recall = TP / (TP + FN) if TP + FN != 0 else 0.
        Specificity = TN / (TN + FP) if TN + FP != 0 else 0.
        F1 = (2 * Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.

        return {
            "Accuracy": acc,
            "micro_Precision": Precision,
            "micro_Recall": Recall,
            "micro_Specificity": Specificity,
            "micro_F1 Score": F1
        }

    def overall_summary_macro(self):
        total_precision = 0
        total_recall = 0
        total_specificity = 0

        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)

        # calculate P,R,S,F1
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            class_precision = TP / (TP + FP) if TP + FP != 0 else 0.
            class_recall = TP / (TP + FN) if TP + FN != 0 else 0.
            class_specificity = TN / (TN + FP) if TN + FP != 0 else 0.

            total_precision += class_precision
            total_recall += class_recall
            total_specificity += class_specificity

        # Macro-average precision and recall
        Precision = total_precision / self.num_classes
        Recall = total_recall / self.num_classes
        Specificity = total_specificity / self.num_classes
        F1 = (2 * Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.

        return {
            "Accuracy": acc,
            "macro_Precision": Precision,
            "macro_Recall": Recall,
            "macro_Specificity": Specificity,
            "macro_F1": F1
        }

    def get_matrix(self):
        matrix = self.matrix
        return matrix

    def plot(self, save_path=None):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90, fontsize=8)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=8)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=10)
        plt.ylabel('Predicted Labels', fontsize=10)
        plt.title('Confusion Matrix', fontsize=10)

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black",
                         fontsize=6)
        plt.tight_layout()
        # Check if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def save_confusion_matrix(matrix, path):
    np.savetxt(path, matrix, fmt='%d')


def save_scores(scores, path, folder_name):

    # Check if the file is empty or doesn't exist
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, 'a') as f:
        # Write the header only if the file is empty
        if write_header:
            f.write('Folder\t' + '\t'.join(scores.keys()) + '\n')

        # Write the scores
        f.write(folder_name + '\t' + '\t'.join([f"{value:.4f}" for value in scores.values()]) + '\n')


def plot_and_save_roc_curve(all_fprs, all_mean_tprs, all_macro_aucs, all_folders, save_path):
    plt.figure(figsize=(10, 10))
    # 使用 viridis 颜色映射，你也可以选择其他如 inferno, plasma, magma, cividis 等
    colors = cm.tab20(np.linspace(0, 1, min(len(all_folders), 20)))

    for i, color in enumerate(colors):
        plt.plot(all_fprs[i], all_mean_tprs[i], color=color, lw=2,
                 label='{0} (area = {1:0.2f})'.format(all_folders[i], all_macro_aucs[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    # 修改x和y轴的刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Macro-Average ROC Curve for Identity Recognition through Each Action', fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.savefig(os.path.join(save_path, 'avg_roc.png'), bbox_inches='tight')
    plt.show()


def clear_directory(path):
    """如果目录存在，清除其下的所有文件；如果不存在，创建目录"""
    if os.path.exists(path):
        # 删除目录下的所有文件
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        # 创建目录
        os.makedirs(path)


def main(args, folder_name):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Processing folder: {folder_name},using device: {device}")

    # 动态加载数据和权重
    args.data_path = os.path.join(args.base_data_path, folder_name)
    args.combined_model_weights = os.path.join(args.weights_path, f"{folder_name}.pt")

    _, _, val_images_path, val_images_label = read_split_data_seq(args.data_path, val_rate=0.3)

    img_size = 224
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    # create combined_model
    model = create_model(num_classes=args.feature_dim)
    mlp = MLP(in_channel=768, classes=args.num_classes)
    combined_model = CombinedModel(model, mlp).to(device)
    # load combined_model weights
    assert os.path.exists(args.combined_model_weights), "cannot find {} file".format(args.combined_model_weights)
    combined_model.load_state_dict(torch.load(args.combined_model_weights, map_location=device))
    combined_model.eval()

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)

    all_labels = []
    all_scores = []
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, _, val_labels = val_data
            outputs = combined_model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs_max = torch.argmax(outputs, dim=1)
            confusion.update(outputs_max.to("cpu").numpy(), val_labels.to("cpu").numpy())
            all_labels.extend(val_labels.to("cpu").numpy())
            all_scores.extend(outputs.to("cpu").numpy())

    # plot_multiclass_roc(all_labels, all_scores, args.num_classes)
    all_fpr, mean_tpr, macro_auc=compute_macro_avg_roc_auc(all_labels, all_scores, args.num_classes)
    # 保存混淆矩阵图像和数据,可替换confusion.overall_summary_micro()
    confusion.plot(save_path=os.path.join(args.confusion_figure_path, f"{folder_name}.png"))
    save_confusion_matrix(confusion.get_matrix(), os.path.join(args.confusion_matrix_path, f"{folder_name}.txt"))
    save_scores(confusion.overall_summary_macro(), os.path.join(args.scores_path, "scores.txt"), folder_name)

    return all_fpr, mean_tpr, macro_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--base-data-path', type=str, default="E:/Pycharm Code/Transformers/Data/Range_Base_HI")
    parser.add_argument('--weights-path', type=str, default='./weights_HI')
    # 保存结果路径
    parser.add_argument('--confusion-matrix-path', type=str, default='./runs/confusion_matrix')
    parser.add_argument('--confusion-figure-path', type=str, default='./runs/confusion_figure')
    parser.add_argument('--roc-figure-path', type=str, default='./runs/roc_figure')
    parser.add_argument('--scores-path', type=str, default='./runs/scores')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    # 创建保存混淆矩阵和分数的文件夹
    clear_directory(opt.confusion_matrix_path)
    clear_directory(opt.confusion_figure_path)
    clear_directory(opt.scores_path)
    clear_directory(opt.roc_figure_path)

    all_fprs = []
    all_mean_tprs = []
    all_macro_aucs = []
    all_folders = []

    # 遍历Range_Base_HI文件夹下的所有文件夹
    for folder in os.listdir(opt.base_data_path):
        if os.path.isdir(os.path.join(opt.base_data_path, folder)):
            fpr, mean_tpr, macro_auc = main(opt, folder)

            all_fprs.append(fpr)
            all_mean_tprs.append(mean_tpr)
            all_macro_aucs.append(macro_auc)
            all_folders.append(folder)

    plot_and_save_roc_curve(all_fprs, all_mean_tprs, all_macro_aucs, all_folders, opt.roc_figure_path)


