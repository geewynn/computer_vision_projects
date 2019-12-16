from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(x, y):
    err = np.sum((x.astype('float') - y.astype('float'))** 2)
    err = err/float(x.shape[0] * y.shape[0])

    return err

def compare_images(x, y, title):
    m = mse(x, y)
    s = measure.compare_ssim(x, y)

    fig = plt.figure(title)
    plt.suptitle('MSE: %.2f, SSIM: %.2f' % (m,s))

    ax = fig.add_subplot(1,2,1)
    plt.imshow(x, cmap=plt.cm.gray)
    plt.axis('off')

    ax = fig.add_subplot(1,2,1)
    plt.imshow(x, cmap=plt.cm.gray)
    plt.axis('off')

    plt.show()

original = cv2.imread('1.jpg')
contrast = cv2.imread('2.jpg')
shopped = cv2.imread('3.jpg')

original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

fig = plt.figure('images')
images = ('original', original), ('contrast', contrast), ('photoshop', shopped)
for i , (name, image) in enumerate(images):
    ax = fig.add_subplot(1,3,i+1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')

plt.show()

compare_images(original, original, 'o vs o')
compare_images(original, contrast, 'o vs c')
compare_images(original, shopped, 'o vs p')