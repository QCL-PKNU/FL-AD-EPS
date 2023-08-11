import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from sklearn.metrics import mean_squared_error

from utils import *


if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['USAD']: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 10
		e = epoch + 1
		start = time()

		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
			
		end = time()
		print(color.BOLD+'Training time: '+"{:10.4f}".format(end-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	### Plot curves
	if not args.test:
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	true_and_prediction(testO, y_pred, f'{args.name}')

	print("RMSE Loss: " + str(mean_squared_error(testO[:, 1], y_pred[:, 1], squared=False)))

	# Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls)
		preds.append(pred)
		df = pd.concat([df, pd.DataFrame(result, index=[i])])
		
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	pprint(result)
