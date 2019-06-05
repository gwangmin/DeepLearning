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

