import matplotlib.pyplot as plt

def plot_3d_vectors(col_vectors, axes_settings = None, **kwargs):
    """
    Plot 3d vectors

    @param col_vectors: 2d array of 3d column vectors
    @param axes_settings: Settings to pass to matplotlib axis set method
    @param **kwargs: Additional args to pass to matplotlib scatter function
    """
    fig, axes = plt.subplots(2,2)    
    axes[1][1].axis('square')
    axes[1][1].axis('off')
    axes = [axes[0][0], axes[0][1], axes[1][0]]
#    fig.subplots_adjust(hspace=0)
    
    axes_indicies = [(0,1), (0,2), (1,2)]
    axes_labels = [('x','y'), ('x', 'z'), ('y', 'z')]

    for (index1, index2), (xlabel, ylabel), ax in zip(axes_indicies, axes_labels, axes):
        ax.scatter(col_vectors[index1,:], col_vectors[index2, :], **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if axes_settings != None:
            ax.set(axes_settings)

        
            
        ax.axis('square')
        
    plt.tight_layout(w_pad=-8)
    plt.show()
