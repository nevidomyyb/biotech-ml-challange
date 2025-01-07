from data_processing import create_dataframe
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def view_tsne():
    df = create_dataframe()
    X = np.vstack(df['embedding'])
    y = df['syndrome_id'].astype('category').values
    tsne = TSNE(n_components=2, random_state=10)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='bright')
    plt.title("Visualization of t-NSE of Biotech Data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Syndrome ID")
    plt.show()
    # tsne_embeddings = tsne.fit_transform(X)

    # # Plot the 2D t-SNE embeddings
    # plt.figure(figsize=(12, 8))
    # for syndrome_id in np.unique(y):
    #     plt.scatter(tsne_embeddings[y == syndrome_id, 0],
    #                 tsne_embeddings[y == syndrome_id, 1],
    #                 label=syndrome_id)
    # plt.legend(loc='best')
    # plt.title('t-SNE Visualization of Embeddings Colored by Syndrome ID')
    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.show()


view_tsne()