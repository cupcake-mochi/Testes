import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from validate import ARTS
from ocean import ocean_mask
def land_core(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); Hh,S,V=[hsv[...,i].astype(int) for i in range(3)]
    oc=ocean_mask(im)
    nonoc=cv2.morphologyEx((1-oc).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,l,s,_=cv2.connectedComponentsWithStats(nonoc)
    keep=np.zeros_like(nonoc)
    for i in range(1,n):
        if s[i][4]>400: keep|=(l==i).astype(np.uint8)
    return keep
def intended(H,W):
    z=np.zeros((H,W),np.uint8)
    for (x0,y0,x1,y1) in ARTS: z[max(0,y0-12):y1+12,max(0,x0-12):x1+12]=1
    for (x0,y0,x1,y1) in [(1212,70,1246,92),(1246,86,1272,106),(1176,48,1220,88)]: z[y0-6:y1+6,x0-6:x1+6]=1
    P=np.load('M_border.npy'); Mh=np.load('M_halo.npy')
    # border band & labels are intended only where they were not real land (LP)
    LP=np.load('M_landprotect.npy')
    z|=(P&(1-cv2.erode(LP,np.ones((11,11),np.uint8))))|Mh
    return z
def damage(a_path,b_path,inner=30):
    a=cv2.imread(a_path); b=cv2.imread(b_path); H,W=a.shape[:2]
    la=land_core(a); diff=np.abs(a.astype(int)-b.astype(int)).sum(2)>60
    dm=(la>0)&diff&(intended(H,W)==0)
    roi=np.zeros((H,W),bool); roi[inner:H-inner,inner:W-inner]=True
    dm&=roi
    dm=cv2.morphologyEx(dm.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,l,s,c=cv2.connectedComponentsWithStats(dm)
    return dm,[(int(c[i][0]),int(c[i][1]),int(s[i][4])) for i in range(1,n) if s[i][4]>30]
if __name__=='__main__':
    dm,bl=damage(sys.argv[1],sys.argv[2])
    print('V6 terra danificada (px):',int(dm.sum()),'blobs:',bl)
