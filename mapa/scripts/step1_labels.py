import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from lama import *
im=cv2.imread('orig.png'); H,W=im.shape[:2]
f=im.astype(np.float32); mx=f.max(2); mn=f.min(2)
dark=((mx<70)&(mx-mn<30)).astype(np.uint8)
tbox={'lumenos':(218,68,562,118),'nordgard':(993,32,1178,80),'santa':(663,426,866,472),'aegis':(108,506,404,558),'nerashi':(952,606,1272,654)}
YY,XX=np.mgrid[:H,:W]
T=np.zeros((H,W),np.uint8); R=np.zeros((H,W),np.float32); ells={}
for k,(x0,y0,x1,y1) in tbox.items():
    reg=np.zeros((H,W),np.uint8); reg[y0:y1,x0:x1]=1
    t=cv2.dilate(dark*reg,np.ones((3,3),np.uint8)); ys,xs=np.nonzero(t)
    t=cv2.dilate(t,np.ones((5,5),np.uint8)); T|=t
    cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2; ax,ay=(xs.max()-xs.min())/2,(ys.max()-ys.min())/2
    ells[k]=(cx,cy,ax,ay)
    d=np.sqrt(((XX-cx)/(ax+70))**2+((YY-cy)/(ay+45))**2)
    R=np.maximum(R,(d<=1).astype(np.float32))
np.save('T_text.npy',T)
# text-free estimate for low-pass
est=cv2.inpaint(im,T,5,cv2.INPAINT_TELEA).astype(np.float32)
sig=6
L=cv2.GaussianBlur(est,(0,0),sig)
# baseline: low-pass of ocean outside halo regions, interpolated inside via normalized conv with big sigma, iterated
out_w=(1-R)
B=L.copy()
for s in [60,40,25]:
    num=cv2.GaussianBlur(L*out_w[...,None],(0,0),s); den=cv2.GaussianBlur(out_w,(0,0),s)[...,None]+1e-6
    B=np.where(R[...,None]>0,num/den,L)
# estimate halo color per pixel ratio: regress over high-alpha zone
Wc=np.array([float(x) for x in sys.argv[1].split(',')],np.float32) if len(sys.argv)>1 else np.array([214,232,236],np.float32)
a=((L-B)/(Wc-B+1e-3)).mean(2)
a=np.clip(a,0,0.92)*R
a=cv2.GaussianBlur(a,(0,0),4)
np.save('alpha.npy',a)
corr=(f-a[...,None]*Wc)/(1-a[...,None])
corr=np.clip(corr,0,255).astype(np.uint8)
cv2.imwrite('s1_dehalo.png',corr)
cv2.imwrite('crops/alpha.png',(a/ max(a.max(),1e-3)*255).astype(np.uint8))
print('alpha max',a.max())
# inpaint: text + core where alpha high (detail destroyed)
mask=cv2.dilate(T,np.ones((9,9),np.uint8))
np.save('M_labels.npy',mask)
out=corr.copy()
for k,(cx,cy,ax,ay) in ells.items():
    # native 512 window(s) covering the label
    w=int(2*ax+90)
    if w<=500:
        out=inpaint_region(out,mask,(int(cx-256),int(cy-256),int(cx+256),int(cy+256)))
    else:
        for xs_ in [cx-ax-45, cx+ax+45-512]:
            out=inpaint_region(out,mask*(np.abs(XX-(xs_+256))<230).astype(np.uint8),(int(xs_),int(cy-256),int(xs_+512),int(cy+256)))
        out=inpaint_region(out,mask,(int(cx-256),int(cy-256),int(cx+256),int(cy+256)))
cv2.imwrite('s1_nolabels.png',out)
