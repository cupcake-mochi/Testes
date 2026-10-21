import numpy as np, cv2, sys
src,dst=sys.argv[1],sys.argv[2]
im=cv2.imread(src).astype(np.float32); ref=cv2.imread('s1_dehalo.png').astype(np.float32)
big=np.load('M_bigland.npy')
H,W=big.shape
box=np.zeros((H,W),np.uint8); box[120:300,25:140]=1   # Lumenos west coast
r=cv2.dilate(big,np.ones((9,9),np.uint8))&box
a=cv2.GaussianBlur(r.astype(np.float32),(0,0),2.5); a=np.maximum(a,cv2.erode(r,np.ones((5,5),np.uint8)).astype(np.float32))[...,None]
out=ref*a+im*(1-a)
cv2.imwrite(dst,np.clip(out,0,255).astype(np.uint8))
