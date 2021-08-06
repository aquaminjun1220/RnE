import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

losses = pd.read_csv("C:/Users/aquam/Documents/GitHub/RnE/model/history.csv", names=["loss", "val_loss"])
print(losses["loss"])

plt.plot(losses['loss'], '-b')
plt.plot(losses['val_loss'], '-r')
plt.legend(['loss', 'val_loss'])
plt.title("loss / val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()