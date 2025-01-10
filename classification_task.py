import numpy as np
import pandas as pd
import matplotlib .pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  StratifiedKFold
from data_processing import create_dataframe
from sklearn.metrics import  precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def classify_knn(metric, x, y):
    
    results = []
    for k in range(1,16):
        precision_ = []
        f1_ = []
        recall_ = []
        
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
            
            precision_.append(precision)
            recall_.append(recall)
            f1_.append(f1)
            
            y_true.extend(y_test)
            y_proba.extend(y_proba_f)
            
        avg_prec = sum(precision_) / len(precision_)
        avg_recall = sum(recall_) / len(recall_)
        avg_f1 = sum(f1_) / len(f1_)
        
        results.append({
            "k": k,
            "precision": avg_prec,
            "recall": avg_recall,
            "f1": avg_f1
        })
    
    return pd.DataFrame(results), y_true, y_proba
        
        
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
    return pd.DataFrame(auc_), axis
    

if __name__ == "__main__":
    df = create_dataframe()
    X = np.vstack(df['embedding'])
    y = df['syndrome_id'].astype('category').values
    euclidean_df, y_true, y_proba  = classify_knn('euclidean', X, y)
    cosine_df, y_true_cosine, y_proba_cosine = classify_knn('cosine', X, y)
    
    figure, (ax1, ax2) = plt.subplots(1,2)
    
    
    euclidean_auc, axis = compute_roc_auc(y, y_proba, "Euclidean", ax1)
    print(euclidean_df)
    print(euclidean_auc)
    
    cosine_auc, axis = compute_roc_auc(y, y_proba_cosine, "Cosine", ax2)
    print(cosine_df)
    print(cosine_auc)
    
    plt.show()