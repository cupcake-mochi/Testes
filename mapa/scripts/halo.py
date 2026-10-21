import numpy as np, cv2
im=cv2.imread('orig.png')
f=im.astype(int); mx=f.max(2); mn=f.min(2)
dark=((mx<70)&(mx-mn<30)).astype(np.uint8)
pale=((mn>95)&((mx-mn)<90)).astype(np.uint8)
boxes={'lumenos':(185,55,600,140),'nordgard':(975,25,1210,100),'santa':(645,405,885,490),'aegis':(90,500,420,570),'nerashi':(935,595,1290,670)}
M=np.zeros(im.shape[:2],np.uint8)
for k,(x0,y0,x1,y1) in boxes.items():
    reg=np.zeros_like(M); reg[y0:y1,x0:x1]=1
    t=cv2.dilate(dark*reg,np.ones((7,7),np.uint8))
    h=cv2.morphologyEx(pale*reg,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    M|=cv2.dilate(((t+h)>0).astype(np.uint8),np.ones((9,9),np.uint8))
np.save('M_labels_raw.npy',M)
for n,(x0,y0,x1,y1) in {'lum':(170,45,620,150),'nord':(960,15,1230,110),'santa':(630,395,900,505),'aeg':(80,490,430,585),'ner':(920,585,1300,690)}.items():
    c=im[y0:y1,x0:x1].copy(); m=M[y0:y1,x0:x1]>0
    c[m]=(0.6*c[m]+[0,0,100]).astype(np.uint8)
    cv2.imwrite(f'crops/hz_{n}.png',cv2.resize(c,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
