
import numpy as np
import cv2
from sklearn.metrics import roc_curve
import torch
import subprocess
import os


def setGPU():
    freeGpu = subprocess.check_output(
        'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)

    if len(freeGpu) == 0:
        print ('No free GPU available!')
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()

    a = torch.cuda.FloatTensor(10)
    return int(freeGpu.strip())




def collage(data, normSamples=False):
    images = [img for img in data.transpose(0, 2, 3, 1)]
    if normSamples:
        for img in images:
            img += img.min()
            img /= img.max()

    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)
    #collage -= collage.min()
    #collage = collage / np.absolute(collage).max() * 256
    return collage


def computeEER(fpr, fnr):
    pos = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[pos] + fnr[pos]) / 2

def evaluate(dataset, testFunction, testSize=64, testIterations=100):
    tstlA = 0
    tstLoss = 0
    for i in range(testIterations):
        img1, img2 = dataset.getBatch(testSize)
        loss, accuracy = testFunction(img1, img2)
        tstlA += accuracy
        tstLoss += loss

    loss = tstLoss / testIterations
    accuracy = tstlA / testIterations
    return  loss, accuracy
