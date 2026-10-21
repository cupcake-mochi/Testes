import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from ocean import *
dh=cv2.imread('s1_dehalo.png'); a=np.load('alpha.npy'); Mh=np.load('M_halo.npy')
H,W=dh.shape[:2]
oc=ocean_mask(dh)
hsv=cv2.cvtColor(dh,cv2.COLOR_BGR2HSV); Hh,S,V=[hsv[...,i].astype(int) for i in range(3)]
# ---- debris boxes (x0,y0,x1,y1)
debris=[(150,392,236,455),(512,408,618,468),(240,286,300,364),(560,528,636,584),(760,396,822,444),
        (842,550,898,592),(880,650,960,705),(1100,660,1162,704),(970,12,1006,34),(1326,36,1386,68),
        (1214,68,1244,88),(1250,88,1268,104),(1312,90,1340,108),(1420,88,1440,102),
        (100,145,130,175),(1056,88,1080,110)]
D=np.zeros((H,W),np.uint8)
for (x0,y0,x1,y1) in debris:
    reg=np.zeros((H,W),np.uint8); reg[y0:y1,x0:x1]=1
    nonoc=((1-oc)&reg)
    d=cv2.dilate(nonoc,np.ones((11,11),np.uint8))
    d=cv2.morphologyEx(d,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    D|=d&cv2.dilate(reg,np.ones((21,21),np.uint8))
# ---- border parchment
parch=((Hh>=5)&(Hh<=32)&(S>=25)&(S<=170)&(V>=120)).astype(np.uint8)
parch=cv2.morphologyEx(parch,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(parch)
P=np.zeros((H,W),np.uint8)
for i in range(1,n):
    x,y,w,h,ar=st[i]
    if (x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2) and ar>150: P|=(lab==i).astype(np.uint8)
# everything in an edge band that is non-ocean is part of the frame junk
band=np.zeros((H,W),np.uint8); band[:30,:]=1; band[H-30:,:]=1; band[:,:18]=1; band[:,W-18:]=1
P|=((1-oc)&band)
# protect real land masses (large non-ocean components not attached to the parchment)
greenish=((Hh>30)&(Hh<85)&(S>90)&(V<190)).astype(np.uint8)
gm=cv2.morphologyEx(greenish,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
gm=cv2.morphologyEx(gm,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
nl,ll,sl,_=cv2.connectedComponentsWithStats(gm)
LP=np.zeros((H,W),np.uint8)
cent=_[1:] if False else None
for i in range(1,nl):
    x,y,w,h,ar=sl[i]; cy=y+h/2
    if ar>300 and cy<H-75 and cy>75: LP|=(ll==i).astype(np.uint8)
nonoc=cv2.morphologyEx((1-oc).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
for (x0,y0,x1,y1) in [(40,120,730,480),(840,85,1515,525),(110,560,560,960),(650,470,860,630),(800,520,1515,900)]:
    bx=np.zeros((H,W),np.uint8); bx[y0:y1,x0:x1]=1; LP|=nonoc&bx
LP=cv2.dilate(LP,np.ones((11,11),np.uint8))
LP[:,:32]=0; LP[:,W-22:]=0; LP[:30,:]=0; LP[H-30:,:]=0
LP=cv2.dilate(LP,np.ones((9,9),np.uint8))
np.save('M_landprotect.npy',LP)
P=cv2.morphologyEx(P,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
P=cv2.dilate(P,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(85,85)))  # include outline + foam ring
# fill holes inside (enclosed bits)
ff=P.copy(); h_,w_=ff.shape; m2=np.zeros((h_+2,w_+2),np.uint8)
inv=(1-ff).astype(np.uint8); cv2.floodFill(inv,m2,(W//2,H//2),2); P=np.where(inv==2,0,1).astype(np.uint8)
P&=(1-LP)
np.save('M_border.npy',P); np.save('M_debris.npy',D)
mask=(Mh|D|P).astype(np.uint8)
ov=dh.copy(); ov[mask>0]=(0.5*ov[mask>0]+[0,0,127]).astype(np.uint8); cv2.imwrite('crops/ov_all.png',ov)
if len(sys.argv)>1 and sys.argv[1]=='preview': sys.exit()
clean=oc.copy(); clean[mask>0]=0
dist=cv2.distanceTransform(clean,cv2.DIST_L2,5)
V_=hsv[...,2].astype(int)
noanom=cv2.erode(((V_>45)).astype(np.uint8),np.ones((15,15),np.uint8))
donor=((dist>22)&(a<0.02)).astype(np.uint8)&noanom
validlp=((dist>12)&(a<0.08)).astype(np.uint8)
cv2.imwrite('crops/donor.png',donor*255); cv2.imwrite('crops/validlp.png',validlp*255)
G=tone_field(dh,validlp)
cv2.imwrite('crops/G.png',np.clip(G,0,255).astype(np.uint8))
out=transplant(dh,mask,donor,valid_lp=validlp,tile=(130,80),G=G)
cv2.imwrite('s2_clean.png',out)
