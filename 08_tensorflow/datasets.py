from __future__ import print_function

import os
import string
import cv2
import numpy as np
import lmdb
import threading
from torch.autograd import Variable
import torch
import torch.nn.functional as F


def randomTransformMatrix(rotDeg, rx, ry, dx, dy, scale):
    rot = rotDeg*np.pi/180.0

    # Move to rotation point
    A1 = np.matrix([[1, 0, -rx],
                    [0, 1, -ry],
                    [0, 0, 1 ]], dtype=np.float64)

    # Rotation matrix
    R = np.matrix([[np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot),  np.cos(rot), 0],
                    [          0,            0, 1]], dtype=np.float64)

    # Move back
    A2 = np.matrix([[1, 0, +rx],
                    [0, 1, +ry],
                    [0, 0, 1 ]], dtype=np.float64)

    T = np.matrix([[1, 0, dx],
                    [0, 1, dy],
                    [0, 0, 1 ]], dtype=np.float64)

    S = np.matrix([[scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1 ]], dtype=np.float64)

    H = A1 * R * A2 * T * S
    return H


def sample(tensor, power, sdev):
    value = tensor.normal_()
    sign = torch.sign(value)
    value = value.abs_().pow_(power) * sdev
    return value * sign


class DataManipulator(object):
    def __init__(self):
        self.rotRange = 1
        self.rotPosRange = 0.1
        self.posRange = 0.03
        self.scaleRange = 0.02

        self.colorSdev = 0.08
        self.contrastSdev = 0.1
        self.minColorSdev = 0.00
        self.gammaSdev = 0.4
        self.noiseSdev = 0.02

        self.transfGammaSdev = 0.4

    def colorManipulation(self, data):
        # gamma1
        power = 0.5

        gamma = sample(torch.cuda.FloatTensor(data.shape[0], 1, 1, 1), power, self.transfGammaSdev)
        gamma = torch.cuda.FloatTensor([2.0]).pow(gamma)
        data.pow_(gamma)

        # color and contrast
        colorCoef = sample(torch.cuda.FloatTensor(data.shape[0], data.shape[1], 1, 1), power, self.colorSdev)
        contrast = torch.cuda.FloatTensor(data.shape[0], 1, 1, 1).normal_(std=self.contrastSdev).abs_()
        data *= torch.cuda.FloatTensor([2.0]).pow(colorCoef -contrast)

        # additive color
        data += torch.cuda.FloatTensor(data.shape[0], 1, 1, 1).normal_(std=self.minColorSdev).abs_()

        # noise
        if self.noiseSdev > 0:
            noiseSdev = torch.cuda.FloatTensor(data.shape[0], 1, 1, 1).normal_().abs_().pow_(power) * self.noiseSdev
            data += torch.cuda.FloatTensor(*data.shape).normal_() * noiseSdev

        data.clamp_(0, 1.0)

        data.pow_(1.0/gamma)

        # final gamma
        gamma = sample(torch.cuda.FloatTensor(data.shape[0], 1, 1, 1), power, self.gammaSdev)
        gamma = torch.cuda.FloatTensor([2.0]).pow(gamma)
        data.pow_(gamma)

        return data

    def geometricTransform(self, data):
        # return data
        shape = data.shape
        T = np.zeros((shape[0], 2, 3))
        r = 0
        rot = np.random.randn(shape[0]) * self.rotRange
        rx = np.random.randn(shape[0]) * self.rotPosRange
        ry = np.random.randn(shape[0]) * self.rotPosRange
        dx = np.random.randn(shape[0]) * self.posRange
        dy = np.random.randn(shape[0]) * self.posRange
        s = 2 ** (np.random.randn(shape[0]) * self.scaleRange)
        for i in range(shape[0]):
            T[i] = randomTransformMatrix(rot[i], rx[i], ry[i], dx[i], dy[i], s[i])[:2, :]
        T = Variable(torch.cuda.FloatTensor(T))
        grid = F.affine_grid(T, shape)
        data = F.grid_sample(data, grid)
        return data.data

    def transform(self, data):

        data = self.geometricTransform(data)
        data = self.colorManipulation(data)
        return data


def border(img, resolution, reptype=cv2.BORDER_REPLICATE):
    out = np.zeros(resolution, dtype=np.uint8)
    img = img[:resolution[0], :resolution[1]]
    out[0:img.shape[0], 0:img.shape[1]] = img
    return out


class PairDatasetAVX(object):
    def __init__(self, listFile, characterFile,
                 path, resolution, randomized=True):
        self.path = path
        self.resolution = resolution
        self.randomized = randomized
        self.toGo = np.zeros(0)

        self.imageNames = []
        self.gt = []
        with open(listFile, 'r') as f:
            for line in f:
                words = line.strip().split()
                self.imageNames.append(words[0])
                self.gt.append(words[1])
        with open(characterFile, 'r') as f:
            chars = [line[0] for line in f]
        print(len(chars), chars)
        self.chars = chars
        fromChar = []
        toChar = []

        for i, c in enumerate(chars):
            fromChar.append(c)
            toChar.append(chr(i))

        trans = string.maketrans(''.join(fromChar), ''.join(toChar))

        for i in range(len(self.gt)):
            self.gt[i] = [ord(x) for x in self.gt[i].translate(trans)]

        self.images = np.zeros(
            (len(self.imageNames), 1, self.resolution[0], self.resolution[1]),
            dtype=np.uint8)

        for i in range(len(self.imageNames)):
            img = cv2.imread(
                os.path.join(path, self.imageNames[i]), 0)
            img = border(img, self.resolution, reptype=cv2.BORDER_DEFAULT)
            self.images[i, 0, :img.shape[0], :img.shape[1]] = img

        self.gt = [np.asarray(s) for s in self.gt]
        self.gt = np.stack(self.gt).astype(np.int32)

    def getBatch(self, count):
        if self.toGo.size < count:
            if self.randomized:
                self.toGo = np.random.permutation(self.images.shape[0])
            else:
                self.toGo = np.arange(self.images.shape[0])

        selectedId = self.toGo[:count]
        self.toGo = self.toGo[count:]

        images = self.images[selectedId]
        gt = self.gt[selectedId]

        images = images.astype(np.float32) / 256.0

        return images, gt

