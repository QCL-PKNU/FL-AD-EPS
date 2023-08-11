import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	# Export y_true and y_pred as cvs file
	np.savetxt(f'plots/{name}/y_true.csv', y_true, delimiter=',')
	np.savetxt(f'plots/{name}/y_pred.csv', y_pred, delimiter=',')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()


def true_and_prediction(y_true, y_pred, name):
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	np.savetxt(f'plots/{name}/y_true.csv', y_true, delimiter=',')
	np.savetxt(f'plots/{name}/y_pred.csv', y_pred, delimiter=',')
	pdf = PdfPages(f'plots/{name}/true_and_prediction.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p = y_true[2000:5000, dim], y_pred[2000:5000, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		ax1.plot(y_t, linewidth=0.2, label='True')
		ax1.plot(y_p, '-', alpha=0.6, linewidth=0.3, label='Predicted')
		pdf.savefig(fig)
		plt.close()
	pdf.close()