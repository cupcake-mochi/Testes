import numpy as np, cv2
im=cv2.imread('orig.png').astype(int)
b,g,r=im[...,0],im[...,1],im[...,2]
mx=np.maximum(np.maximum(r,g),b); mn=np.minimum(np.minimum(r,g),b)
dark=((mx<70)&(mx-mn<30)).astype(np.uint8)
regions={'lumenos':(200,60,590,125),'nordgard':(980,30,1190,90),'santa':(650,420,880,480),'aegis':(90,505,420,560),'nerashi':(940,600,1290,660)}
for k,(x0,y0,x1,y1) in regions.items():
    sub=dark[y0:y1,x0:x1]; ys,xs=np.nonzero(sub)
    print(k, xs.min()+x0, ys.min()+y0, xs.max()+x0, ys.max()+y0, sub.sum())
    # halo: brightness above local ocean
    lum=im[...,::-1].mean(2)
