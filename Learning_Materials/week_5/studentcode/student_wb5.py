# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here
    data = np.genfromtxt(datafile_name, delimiter=',')

    k_model = KMeans(n_clusters=K)
    k_model.fit(data)
    labels = k_model.labels_

    features = data.shape[1]

    fig, axs = plt.subplots(features, features, sharex='col')

    for i in range(features):
        for j in range(features):
            if i != j:
                axs[i,j].scatter(data[:,j], data[:,i], c=labels, cmap='tab10')
                axs[i,j].set_ylim(np.min(data[:,i]), np.max(data[:,i]))
            else:
                bins = np.linspace(np.min(data[:,i]), np.max(data[:,i]), 11)
                for k in range(K):
                    axs[i,i].hist(data[labels==k, i], bins=bins, color=plt.cm.tab10(k), alpha=0.5)

    for i in range(features):
        axs[i,0].set_ylabel(feature_names[i])
    for j in range(features):
        axs[-1,j].set_xlabel(feature_names[j])

    username = "s4-gauchan"
    fig.suptitle(f"Visualisation of {K} clusters by {username}")

    fig.savefig('myVisualisation.jpg')
    
    return fig, axs    
    # <==== insert your code above here
