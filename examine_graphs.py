import numpy as np
import matplotlib.pyplot as plt

train_loss, val_loss = np.loadtxt("gnn_loss_vals_0")
percent = 80
train_max = np.max(train_loss[(100 - percent):])
val_max = np.max(val_loss[(100 - percent):])
# train_loss = train_loss[(100 - percent):]
# val_loss = val_loss[(100 - percent):]

plt.plot(train_loss, color='red', label="train loss")
plt.plot(val_loss, color='blue', label="val loss")
plt.xlim([100 - percent, 100])
plt.ylim([0, max(train_max, val_max) * 2])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.savefig(f"gnn_loss_plt_percent{percent}.png")