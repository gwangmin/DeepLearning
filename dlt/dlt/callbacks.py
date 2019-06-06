"""
This file defines callbacks.
"""

import keras.callbacks as callbacks

class HistoryCallback(callbacks.Callback):
    """
    Callback for graph
    Usage: his = HistoryCallback()
           his.init()
           model.fit(..., callbacks=[..., his])
    """
    
    def init(self):
        """
        Init this object
        """
        self.history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
        
    def on_epoch_end(self, batch, logs={}):
        """
        Callback
        """
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['acc'].append(logs.get('acc'))
        self.history['val_acc'].append(logs.get('val_acc'))
