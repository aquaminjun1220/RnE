import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import random

def plot_hist(path):
    legend = []
    plt.figure(figsize=(8,6))
    plt.rc('font', size=12)
    for model in os.listdir(path):
        hist = path + '/' + model + '/' + 'history.csv'
        losses = pd.read_csv(hist, names=["loss", "val_loss"])
        r = random.random()
        g = random.random()
        b = random.random()
        color = (r, g, b)
        plt.plot(losses['loss'], c=color)
        r = random.random()
        g = random.random()
        b = random.random()
        color = (r, g, b)        
        plt.plot(losses['val_loss'], c=color)
        legend.append('loss({model})'.format(model=model))
        legend.append('val_loss({model})'.format(model=model))
    plt.title("loss / val_loss ")
    plt.legend(legend)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    plot_hist(path=args.path)
