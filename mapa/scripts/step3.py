import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from ocean import *
src=sys.argv[1]; dst=sys.argv[2]
im=cv2.imread(src); H,W=im.shape[:2]
oc=ocean_mask(im)
nonoc=cv2.morphologyEx((1-oc).astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
M=np.zeros((H,W),np.uint8)
# a) left blotch & generic edge junk in left 70px
reg=np.zeros((H,W),np.uint8); reg[100:200,0:46]=1
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); S=hsv[...,1].astype(int)
pale=((S<70)|(oc==0)).astype(np.uint8)&reg
M|=cv2.dilate(cv2.morphologyEx(pale,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)),np.ones((15,15),np.uint8))&cv2.dilate(reg,np.ones((3,3),np.uint8))
# b) small floating specks in Nordgard north waters (not attached to big land)
reg=np.zeros((H,W),np.uint8); reg[55:105,1200:1450]=1
n,lab,st,cen=cv2.connectedComponentsWithStats(nonoc)
for i in range(1,n):
    x,y,w,h,a=st[i]
    if a<120 and reg[int(cen[i][1]),int(cen[i][0])]:
        M|=cv2.dilate((lab==i).astype(np.uint8),np.ones((9,9),np.uint8))
extra=[tuple(map(int,b.split(','))) for b in sys.argv[3:]]
for (x0,y0,x1,y1) in extra:
    r=np.zeros((H,W),np.uint8); r[y0:y1,x0:x1]=1
    odd=((nonoc>0)|(S<75)).astype(np.uint8)&r
    M|=cv2.dilate(odd,np.ones((11,11),np.uint8))&cv2.dilate(r,np.ones((15,15),np.uint8))
sys.path.insert(0,'scripts')
from landcheck import land_core
lc=land_core(cv2.imread('s1_dehalo.png'))
n_,l_,s_,_=cv2.connectedComponentsWithStats(lc)
big=np.zeros_like(lc)
for i in range(1,n_):
    if s_[i][4]>2000: big|=(l_==i).astype(np.uint8)
M&=(1-cv2.dilate(big,np.ones((9,9),np.uint8)))
np.save('M_bigland.npy',big)
print('mask px',M.sum())
clean=oc.copy(); clean[M>0]=0
dist=cv2.distanceTransform(clean,cv2.DIST_L2,5)
a=np.load('alpha.npy')
donor=((dist>22)&(a<0.02)).astype(np.uint8)
validlp=((dist>12)).astype(np.uint8)
G=tone_field(im,validlp)
out=transplant(im,M,donor,valid_lp=validlp,tile=(130,80),G=G) if M.sum() else im.copy()
# c) low-frequency tone correction inside enlarged label zones (ocean only)
YY,XX=np.mgrid[:H,:W]
tb=[(218,68,562,118),(993,32,1178,80),(663,426,866,472),(108,506,404,558),(952,606,1272,654)]
wgt=np.zeros((H,W),np.float32)
for (x0,y0,x1,y1) in tb:
    cx,cy,ax,ay=(x0+x1)/2,(y0+y1)/2,(x1-x0)/2+80,(y1-y0)/2+55
    d=np.sqrt(((XX-cx)/ax)**2+((YY-cy)/ay)**2)
    wgt=np.maximum(wgt,np.clip((1.15-d)/0.3,0,1))
oc2=ocean_mask(out); dist2=cv2.distanceTransform(oc2,cv2.DIST_L2,5)
ocw=np.clip((dist2-3)/8,0,1)
valid_out=(dist2>12)&(wgt<0.05)
G2=tone_field(out,valid_out.astype(np.uint8))
lpo=lowpass_masked(out,(dist2>4),10)
corr=(G2-lpo)*(wgt*ocw)[...,None]
out=np.clip(out.astype(np.float32)+corr,0,255).astype(np.uint8)
cv2.imwrite(dst,out)
