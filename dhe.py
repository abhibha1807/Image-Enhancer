import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import imageio
import scipy, scipy.misc, scipy.signal
import cv2
import sys
from os import listdir
from os.path import isfile, join
def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    #ch = 1
    Img = np.zeros((hei+4, wid+4, ch))
    for i in range(ch):
        Img[:,:,i] = np.pad(img[:,:,i], (2,2), 'edge')
        #Img[:,:] = np.pad(img[:,:], (2,2), 'edge')

    hsv = (matplotlib.colors.rgb_to_hsv(Img))
    hsv[:,:,0] = hsv[:,:,0] * 255
    hsv[:,:,1] = hsv[:,:,1] * 255
    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
    hsv = hsv.astype(np.uint8).astype(np.float64)
    fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    fv = fh.conj().T
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    I = hsv[:,:,2]

    dIh = scipy.signal.convolve2d(I, np.rot90(fh, 2), mode='same')
    dIv = scipy.signal.convolve2d(I, np.rot90(fv, 2), mode='same')
    dIh[dIh==0] = 0.00001
    dIv[dIv==0] = 0.00001
    dI = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
    di = dI[2:hei+2,2:wid+2]
    
    dSh = scipy.signal.convolve2d(S, np.rot90(fh, 2), mode='same')
    dSv = scipy.signal.convolve2d(S, np.rot90(fv, 2), mode='same')
    dSh[dSh==0] = 0.00001
    dSv[dSv==0] = 0.00001
    dS = np.sqrt(dSh**2+dSv**2).astype(np.uint32)
    ds = dS[2:hei+2,2:wid+2]

    
    h = H[2:hei+2,2:wid+2]
    s = S[2:hei+2,2:wid+2]
    i = I[2:hei+2,2:wid+2].astype(np.uint8)
    
    Imean = scipy.signal.convolve2d(I,np.ones((5,5))/25, mode='same')
    Smean = scipy.signal.convolve2d(S,np.ones((5,5))/25, mode='same')
    
    Rho = np.zeros((hei+4,wid+4))
    for p in range(2,hei+2):
        for q in range(2,wid+2):
            tmpi = I[p-2:p+3,q-2:q+3]
            tmps = S[p-2:p+3,q-2:q+3]
            corre = np.corrcoef(tmpi.flatten('F'),tmps.flatten('F'))
            Rho[p,q] = corre[0,1]
    
    rho = np.abs(Rho[2:hei+2,2:wid+2])
    rho[np.isnan(rho)] = 0
    rd = (rho*ds).astype(np.uint32)
    Hist_I = np.zeros((256,1))
    Hist_S = np.zeros((256,1))
    
    for n in range(0,255):
        temp = np.zeros(di.shape)
        temp[i==n] = di[i==n]
        Hist_I[n+1] = np.sum(temp.flatten('F'))
        temp = np.zeros(di.shape)
        temp[i==n] = rd[i==n]
        Hist_S[n+1] = np.sum(temp.flatten('F'))

    return Hist_I, Hist_S

def dhe(img, alpha=0.5):
    
    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha*hist_s + (1-alpha)*hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)
    
    hsv = matplotlib.colors.rgb_to_hsv(img)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    i = hsv[:,:,2].astype(np.uint8)
    
    c = hist_cum / hist_sum
    s_r = (c * 255)
    i_s = np.zeros(i.shape)
    for n in range(0,255):
        i_s[i==n] = s_r[n+1]/255.0
    i_s[i==255] = 1
    hsi_o = np.stack((h,s,i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)
    
    result = result * 255
    result[result>255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def main():
    # img_name = sys.argv[1]
    #filename1 = '/abhibha-volume/PCDARTS-cifar10/data/train/NORMAL/'
    #filename2 = '/abhibha-volume/PCDARTS-cifar10/data/train/PNEUMONIA/'
    #filename3 = '/abhibha-volume/PCDARTS-cifar10/data/test/NORMAL/'
    #filename4 = '/abhibha-volume/PCDARTS-cifar10/data/test/PNEUMONIA/'
    
    #filename5 = '/abhibha-volume/PCDARTS-cifar10/data_copy/train/NORMAL/'
    filename6 = '/abhibha-volume/PCDARTS-cifar10/data_copy/train/PNEUMONIA/'
    # filename7 = '/abhibha-volume/PCDARTS-cifar10/data_copy/test/NORMAL/'
    # filename8 = '/abhibha-volume/PCDARTS-cifar10/data_copy/test/PNEUMONIA/'
    
    #onlyfiles1 = [f for f in listdir(filename1) if isfile(join(filename1, f))]
    #onlyfiles2 = [f for f in listdir(filename2) if isfile(join(filename2, f))]
    #onlyfiles3 = [f for f in listdir(filename3) if isfile(join(filename3, f))]
    #onlyfiles4 = [f for f in listdir(filename4) if isfile(join(filename4, f))]

    #onlyfiles5 = [f for f in listdir(filename5) if isfile(join(filename5, f))]
    onlyfiles6 = [f for f in listdir(filename6) if isfile(join(filename6, f))]
    # onlyfiles7 = [f for f in listdir(filename7) if isfile(join(filename7, f))]
    # onlyfiles8 = [f for f in listdir(filename8) if isfile(join(filename8, f))]

    #print(onlyfiles)
    # c=0
    # for i in onlyfiles1:
    #     c=c+1
    #     if c > 120:
    #         towrite = './cleaned_train/NORMAL/'
    #         img_name = filename1 + i
    #         img = imageio.imread(img_name)
    #         img_ = cv2.imread(img_name)
    #         #plt.imshow(img_)
    #         l=img.shape[0]
    #         w=img.shape[1]
            
    #         img = img_.reshape(l,w,3)
    #         print(img.shape)
    #         result = dhe(img)
    #         #plt.imshow(result)
    #         #plt.show()
    #         cv2.imwrite(towrite+i, result)
    
    # for i in onlyfiles2:
    #     towrite = './cleaned_train/PNEUMONIA/'
    #     img_name = filename2 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)

    # for i in onlyfiles3:
    #     towrite = './cleaned_test/NORMAL/'
    #     img_name = filename3 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)
    
    # for i in onlyfiles4:
    #     towrite = './cleaned_test/PNEUMONIA/'
    #     img_name = filename4 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)
    
    # for i in onlyfiles5:
    #     towrite = './cleaned_train_copy/NORMAL/'
    #     img_name = filename5 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)
    
    for i in onlyfiles6:
        towrite = './cleaned_train_copy/PNEUMONIA/'
        img_name = filename6 + i
        img = imageio.imread(img_name)
        img_ = cv2.imread(img_name)
        #plt.imshow(img_)
        l=img.shape[0]
        w=img.shape[1]
        
        img = img_.reshape(l,w,3)
        print(img.shape)
        result = dhe(img)
        #plt.imshow(result)
        #plt.show()
        cv2.imwrite(towrite+i, result)

    # for i in onlyfiles7:
    #     towrite = './cleaned_test_copy/NORMAL/'
    #     img_name = filename7 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)
    
    # for i in onlyfiles8:
    #     towrite = './cleaned_test_copy/PNEUMONIA/'
    #     img_name = filename8 + i
    #     img = imageio.imread(img_name)
    #     img_ = cv2.imread(img_name)
    #     #plt.imshow(img_)
    #     l=img.shape[0]
    #     w=img.shape[1]
        
    #     img = img_.reshape(l,w,3)
    #     print(img.shape)
    #     result = dhe(img)
    #     #plt.imshow(result)
    #     #plt.show()
    #     cv2.imwrite(towrite+i, result)
    
       

if __name__ == '__main__':
    main()
