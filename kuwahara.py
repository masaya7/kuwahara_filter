import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def kuwahara(pic, r=5, resize=False, rate=0.5):  # 元画像、正方形領域の一辺、リサイズするか、リサイズする場合の比率
    h, w, _ = pic.shape
    if resize: pic = cv2.resize(pic, (int(w * rate), int(h * rate)));h, w, _ = pic.shape
    pic = np.pad(pic, ((r, r), (r, r), (0, 0)), "edge")
    ave, var = cv2.integral2(pic)
    ave = (ave[:-r - 1, :-r - 1] + ave[r + 1:, r + 1:] - ave[r + 1:, :-r - 1] - ave[:-r - 1, r + 1:]) / (
            r + 1) ** 2  # 平均値の一括計算
    var = ((var[:-r - 1, :-r - 1] + var[r + 1:, r + 1:] - var[r + 1:, :-r - 1] - var[:-r - 1, r + 1:]) / (
            r + 1) ** 2 - ave ** 2).sum(axis=2)  # 分散の一括計算

    def filt(i, j):
        return np.array([ave[i, j], ave[i + r, j], ave[i, j + r], ave[i + r, j + r]])[(
            np.array([var[i, j], var[i + r, j], var[i, j + r], var[i + r, j + r]]).argmin(axis=0).flatten(),
            j.flatten(),
            i.flatten())].reshape(w, h, _).transpose(1, 0, 2)

    filtered_pic = filt(*np.meshgrid(np.arange(h), np.arange(w))).astype(pic.dtype)  # 色の決定
    return filtered_pic


#画像保存ディレクトリ
os.chdir(r'assets/img/sample.jpg')

# 画像名を入れる
picture = np.array(plt.imread("sample.jpg"))
filtered_pic = kuwahara(pic=picture, r=15, resize=False, rate=0.2)
plt.imshow(filtered_pic)
plt.axis("off")
plt.show()