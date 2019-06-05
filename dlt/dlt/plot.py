"""
This file defines plot funcs.
"""

import matplotlib.pylab as plt

def showLoss(history):
    """
    Show loss graph in train, validation. Based on history.
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','val'], loc=0)
    plt.show()

def showAcc(history):
    """
    Show Accuracy graph in train, validation. Based on history.
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(['train','val'], loc=0)
    plt.show()

def visualizeFilter(filters, size=1):
    """
    Visualize the conv filter(s)
    
    size: 1 or tuple(rows,cols).
    """
    plt.title('Filter(s)')
    if size == 1:
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filters)
    else:
        idx= size[0] * size[1]
        for i in range(idx):
            plt.subplot(*size, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(filters[i])
    plt.tight_layout()
    plt.show()
