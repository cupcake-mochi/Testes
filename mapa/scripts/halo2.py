import numpy as np, cv2
im=cv2.imread('orig.png')
f=im.astype(np.float32); mx=f.max(2); mn=f.min(2)
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
mu=cv2.blur(gray,(9,9)); sd=np.sqrt(np.maximum(cv2.blur(gray*gray,(9,9))-mu*mu,0))
dark=((mx<70)&(mx-mn<30)).astype(np.uint8)
pale=((mn>90)&((mx-mn)<100)).astype(np.uint8)
# land detector: strong edges with dark outlines, or green/brown/sand hues
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
H,S,V=hsv[...,0].astype(int),hsv[...,1].astype(int),hsv[...,2].astype(int)
greenish=((H>20)&(H<85)&(S>60)).astype(np.uint8)
sandy=((H>5)&(H<30)&(S>50)).astype(np.uint8)
tbox={'lumenos':(218,68,562,118),'nordgard':(993,32,1178,80),'santa':(663,426,866,472),'aegis':(108,506,404,558),'nerashi':(952,606,1272,654)}
boxes={'lumenos':(185,55,600,140),'nordgard':(975,25,1210,100),'santa':(645,405,885,490),'aegis':(90,500,420,570),'nerashi':(935,595,1290,670)}
M=np.zeros(im.shape[:2],np.uint8)
for k,(x0,y0,x1,y1) in boxes.items():
    reg=np.zeros_like(M); reg[y0:y1,x0:x1]=1
    tx0,ty0,tx1,ty1=tbox[k]; treg=np.zeros_like(M); treg[ty0:ty1,tx0:tx1]=1
    t=cv2.dilate(dark*treg,np.ones((5,5),np.uint8))
    t=cv2.morphologyEx(t,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    tz=cv2.dilate(t,np.ones((25,25),np.uint8))  # zone near text
    h=pale*reg
    land=cv2.dilate(((greenish|sandy)*reg),np.ones((3,3),np.uint8))
    land=cv2.morphologyEx(land,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    land=cv2.dilate(land,np.ones((7,7),np.uint8))
    h=h&(1-land)
    h=cv2.morphologyEx(h,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    h=cv2.morphologyEx(h,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    # keep halo components touching text zone
    n,lab=cv2.connectedComponents(h)
    keep=np.isin(lab,np.unique(lab[(tz>0)&(h>0)]))&(lab>0)
    ys,xs=np.nonzero(t); cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
    ax,ay=(xs.max()-xs.min())/2+22,(ys.max()-ys.min())/2+12
    YY,XX=np.mgrid[:M.shape[0],:M.shape[1]]
    ell=(((XX-cx)/ax)**2+((YY-cy)/ay)**2<=1).astype(np.uint8)
    edges=cv2.Canny(im,80,200); ed=cv2.blur((edges>0).astype(np.float32),(11,11))
    busy=cv2.dilate((ed>0.12).astype(np.uint8),np.ones((5,5),np.uint8))
    hm=(keep.astype(np.uint8)&ell&(1-busy))
    hm=cv2.morphologyEx(hm,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    hm=cv2.dilate(hm,np.ones((5,5),np.uint8))&(1-land)&ell
    m=(hm|cv2.dilate(t,np.ones((5,5),np.uint8))).astype(np.uint8)
    M|=m
np.save('M_labels.npy',M)
for n,(x0,y0,x1,y1) in {'lum':(170,45,620,150),'nord':(960,15,1230,110),'santa':(630,395,900,505),'aeg':(80,490,430,585),'ner':(920,585,1300,690)}.items():
    c=im[y0:y1,x0:x1].copy(); m=M[y0:y1,x0:x1]>0
    c[m]=(0.6*c[m]+[0,0,100]).astype(np.uint8)
    cv2.imwrite(f'crops/hz_{n}.png',cv2.resize(c,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
