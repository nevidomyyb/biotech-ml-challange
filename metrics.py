import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def create_plot(df: pd.DataFrame, column: str, axis, title: str = ""):
    column_names = column.split('_')
    if len(column_names) ==2 :
        column_ = column_names[1].capitalize()
    else:
        column_ = " ".join(column_names[1:]).capitalize()
    if column_ == "F1":
        column_ = "F1-Score"
    
    
    x = df['k']
    y = df[column]
    
    axis.plot(x, y, label=column_)
    axis.grid()
    
    axis.set_xticks(np.arange(len(x)+1))
    axis.set_title(title)
    axis.set_xlabel('K')
    axis.set_ylabel('Value')
    axis.legend()
    
    
if __name__ == "__main__":
    df_euclidean = pd.read_csv('./euclidean_metrics.csv', sep=';')
    df_cosine = pd.read_csv('./cosine_metrics.csv', sep=';')
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    df_precision = df_euclidean.drop(['avg_recall', 'avg_f1', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_recall = df_euclidean.drop(['avg_precision', 'avg_f1', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_f1 = df_euclidean.drop(['avg_recall', 'avg_precision', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_accuracy = df_euclidean.drop(['avg_recall', 'avg_f1', 'avg_precision', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_top_k = df_euclidean.drop(['avg_recall', 'avg_f1', 'avg_accuracy', 'avg_precision'], inplace=False, axis=1)
    
    df_precision_c = df_cosine.drop(['avg_recall', 'avg_f1', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_recall_c = df_cosine.drop(['avg_precision', 'avg_f1', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_f1_c = df_cosine.drop(['avg_recall', 'avg_precision', 'avg_accuracy', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_accuracy_c = df_cosine.drop(['avg_recall', 'avg_f1', 'avg_precision', 'avg_top_k_accuracy'], inplace=False, axis=1)
    df_top_k_c = df_cosine.drop(['avg_recall', 'avg_f1', 'avg_accuracy', 'avg_precision'], inplace=False, axis=1)
    
    
    create_plot(df_precision, 'avg_precision', ax1)
    create_plot(df_recall, 'avg_recall', ax1)
    create_plot(df_f1, 'avg_f1', ax1)
    create_plot(df_accuracy, 'avg_accuracy', ax1)
    create_plot(df_top_k, 'avg_top_k_accuracy', ax1, "Euclidean Metrics")
    
    create_plot(df_precision_c, 'avg_precision', ax2)
    create_plot(df_recall_c, 'avg_recall', ax2)
    create_plot(df_f1_c, 'avg_f1', ax2)
    create_plot(df_accuracy_c, 'avg_accuracy', ax2)
    create_plot(df_top_k_c, 'avg_top_k_accuracy', ax2, "Cosine Metrics")
    
    
    plt.show()