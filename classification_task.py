import numpy as np
import pandas as pd
import matplotlib .pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  StratifiedKFold
from data_processing import create_dataframe
from sklearn.metrics import  precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

def classify_knn(metric, x, y, top_axis, bottom_axis=None):
    
    results = []
    y_true_top_k = []
    y_proba_top_k = []
    accuracy_top_k = 0
    top_k = 0
    
    y_true_overall = []
    y_proba_overall = []
    for k in range(1,16):
        precision_ = []
        f1_ = []
        recall_ = []
        accuracy_ = []
        
        knn = KNeighborsClassifier(n_neighbors=k,  metric=metric)
        cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        y_true = []
        y_proba = []
        
        for train_i, test_i in cross_validation.split(x, y):
            x_train, x_test = x[train_i], x[test_i]
            y_train, y_test = y[train_i], y[test_i]
            
            knn.fit(x_train, y_train)
            y_pred_f = knn.predict(x_test)
            y_proba_f = knn.predict_proba(x_test)
            
            #IA para debugar e usar "average='weighted'"
            precision = precision_score(y_test, y_pred_f, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_f, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_f, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred_f)
            
            precision_.append(precision)
            recall_.append(recall)
            f1_.append(f1)
            accuracy_.append(accuracy)
            
            y_true.extend(y_test)
            y_proba.extend(y_proba_f)
            
            y_true_overall.extend(y_test)
            y_proba_overall.extend(y_proba_f)
            
        avg_prec = sum(precision_) / len(precision_)
        avg_recall = sum(recall_) / len(recall_)
        avg_f1 = sum(f1_) / len(f1_)
        avg_accuracy = sum(accuracy_) / len(accuracy_)
        if avg_accuracy > accuracy_top_k:
            accuracy_top_k = avg_accuracy
            y_true_top_k = y_true
            y_proba_top_k = y_proba
            top_k = k
        
        results.append({
            "k": k,
            "avg_precision": avg_prec,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_accuracy": avg_accuracy
        })
    auc_df_top_k = compute_roc_auc(y_true_top_k, y_proba_top_k, f"{metric} - TOP K = {top_k}", top_axis)
    auc_df_overall = compute_roc_auc(y_true_overall, y_proba_overall, f"{metric} - Overall", bottom_axis)
    return pd.DataFrame(results), auc_df_top_k, auc_df_overall
        
        
def compute_roc_auc(y, y_proba, metric, axis):
    auc_ = []
    y_bin = label_binarize(y, classes=np.unique(y))
    # IA para ajustar a dimensão do array.
    y_proba = np.array(y_proba)
    y_label = np.unique(y)
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_value = auc(fpr, tpr)
        axis.plot(fpr, tpr, label=f"{y_label[i]} (AUC = {auc_value:.2f})")
        auc_.append({
            "class": y_label[i],
            "auc": auc_value
        })
    
    axis.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"ROC Curve - {metric}")
    axis.legend()
    return pd.DataFrame(auc_)
    

if __name__ == "__main__":
    df = create_dataframe()
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    X = np.vstack(df['embedding'])
    y = df['syndrome_id'].astype('category').values
    euclidean_df, euclidean_auc_top_k_df, euclidean_auc_overall_df  = classify_knn('euclidean', X, y, ax1, ax3)
    cosine_df, cosine_auc_top_k_df, cosine_auc_overall_df = classify_knn('cosine', X, y, ax2, ax4)
    
    euclidean_df.to_csv('./euclidean_metrics.csv', header=True, sep=';', index=False)
    euclidean_auc_top_k_df.to_csv('./euclidean_auc_top_k.csv', header=True, sep=';', index=False)
    euclidean_auc_overall_df.to_csv('./euclidean_auc_overall.csv', header=True, sep=';', index=False)
    
    
    # euclidean_figure, ax1 = plt.subplots(1, 1)
    
    # euclidean_df = euclidean_df.to_dict(orient='list')
    # values_x = euclidean_df['k']
    # del euclidean_df['k']
    # x = np.arange(len(values_x))
    # width = 0.25
    # multiplier = 0
    
    # for attribute, value in euclidean_df.items():
    #     offset = width * multiplier
    #     rects = ax1.bar(x + offset, value, width, label=attribute)
    #     ax1.bar_label(rects, padding=3)
    #     multiplier += 1
    
    # ax1.set_ylabel("Value in %")
    # ax1.set_xlabel("Value of K")
    # ax1.set_xticks(x + width, values_x)
    # ax1.legend(loc='upper left', ncols=4)
    # ax1.set_ylim(0, 250)
    
    print(euclidean_df)
    
    plt.show()