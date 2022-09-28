import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


def activationFunc(x):
    if (x < 0):
        return -1
    else:
        return 1

def updatefig(*args):
    global v
    global w
    global dim
    global step
    for i in range (0,step):
        p = np.random.randint(0,dim**2)
        v[p] = activationFunc(np.dot(w[p],v))
    im.set_array(v.reshape (dim,dim))
    return im,    
    
t = 1000
step = 50
dim = 32
vActivationFunc = np.vectorize(activationFunc)

print('memorising image . . .')


image1 = Image.open('triangle.png').convert('L')
imgArray1 = np.asarray(image1)
dim = imgArray1.shape[0]
imgVec1 = imgArray1.reshape(dim**2)
imgVec1 = imgVec1 - 128 * np.ones(dim**2)
imgVec1 = vActivationFunc(imgVec1)
w1 = np.outer(imgVec1, imgVec1)

image2 = Image.open('cross.png').convert('L')
imgArray2 = np.asarray(image2)
imgVec2 = imgArray2.reshape(dim**2)
imgVec2 = imgVec2 - 128 * np.ones(dim**2)
imgVec2 = vActivationFunc(imgVec2)
w2 = np.outer(imgVec2, imgVec2)

image3 = Image.open('circle.png').convert('L')
imgArray3 = np.asarray(image3)
imgVec3 = imgArray3.reshape(dim**2)
imgVec3 = imgVec3 - 128 * np.ones(dim**2)
imgVec3 = vActivationFunc(imgVec3)
w3 = np.outer(imgVec3, imgVec3)

image4 = Image.open('square.png').convert('L')
imgArray4 = np.asarray(image4)
imgVec4 = imgArray4.reshape(dim**2)
imgVec4 = imgVec4 - 128 * np.ones(dim**2)
imgVec4 = vActivationFunc(imgVec4)
w4 = np.outer(imgVec4, imgVec4)

w = w4

"""
w = np.random.rand(dim**2,dim**2)
w = w - 0.5*np.ones((dim**2,dim**2))
w = vActivationFunc(w)
"""

v = np.random.rand(dim**2)
v = v - 0.5*np.ones(dim**2)
v = vActivationFunc(v)


input("image memorized, would you like to play? >")

fig = plt.figure()
im = plt.imshow(v.reshape(dim,dim), animated=True)

ani = animation.FuncAnimation(fig, updatefig,  blit=True)
plt.show()

    
