import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# the function plots the scatter of the df we want to plot,
# also numpy data (for examplr after scaling) is ok to send with no conversion,
# df_with_target is any df who has the targets variables with no missing values from the df we send to plot,
# and with the same indexes.

def plot_points_scatter(df_to_plot, df_with_target_varaible, title_t):
    pca_model = PCA(n_components=2)
    data_transformed = pca_model.fit_transform(df_to_plot)
    principalDf = pd.DataFrame(data=data_transformed, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df_with_target_varaible[['isMalicious']]], axis=1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(title_t, fontsize = 20)

    targets = [False, True]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['isMalicious'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
