import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
np.seterr(divide='ignore',invalid='ignore')
img=cv.imread('/home/oem/下载/orange.jpg',0)#读入图
f=np.fft.fft2(img,axes=(0,1))
fshift=np.fft.fftshift(f)
res=np.log(np.abs(fshift))#幅度谱
ag=np.angle(fshift)#相位谱

ishift1=np.fft.ifftshift(res)#利用幅度谱逆变换
iimg1=np.fft.ifft2(ishift1)
iimg1=np.abs(iimg1)

ishift=np.fft.ifftshift(res)#利用相位谱逆变换
iimg=np.fft.ifft2(ishift1)
iimg=np.abs(iimg)

ishift2=np.fft.ifftshift(fshift)#整体逆变换
iimg2=np.fft.ifft2(ishift2)
iimg2=np.abs(iimg2)

retval,dst=cv.threshold(img,127,255,cv.THRESH_TOZERO)#设置图像阈值
f2=np.fft.fft2(dst,axes=(0,1))
fshift3=np.fft.fftshift(f2)
res2=np.log(np.abs(fshift3))#幅度谱
ishift3=np.fft.ifftshift(res2)#利用幅度谱逆变换
result=np.fft.ifft2(ishift2)
result=np.abs(result)

plt.subplot(421),plt.imshow(res,'gray'),plt.title('Amplitude spectral transformation')#幅度谱变换
plt.axis('off')
plt.subplot(422),plt.imshow(ag,'gray'),plt.title('Phase spectral transforamtion')#相位谱变换
plt.axis('off')
plt.subplot(423),plt.imshow(dst,'gray'),plt.title('threshold')#设置阈值
plt.axis('off')
plt.subplot(424),plt.imshow(result,'gray'),plt.title('reverse')#傅里叶逆变换
plt.axis('off')
plt.subplot(425),plt.imshow(iimg,'gray'),plt.title('Inverse phase spectral transforamtion')#相位谱逆变换
plt.axis('off')
plt.subplot(426),plt.imshow(iimg1,'gray'),plt.title('Inverse transforamtion of the amplitude spectrum')#幅度谱逆变换
plt.axis('off')
plt.subplot(427),plt.imshow(iimg2,'gray'),plt.title('Global inverse transformation')#整体逆变换
plt.axis('off')
plt.show()

