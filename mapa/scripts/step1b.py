import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from ocean import *
dh=cv2.imread('s1_dehalo.png'); a=np.load('alpha.npy'); T=np.load('T_text.npy')
oc=ocean_mask(dh)
cv2.imwrite('crops/oceanmask.png',oc*255)
land=(1-oc)
landp=cv2.dilate(cv2.morphologyEx(land,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)),np.ones((5,5),np.uint8))
textz=cv2.dilate(T,np.ones((11,11),np.uint8))
mask=(((a>0.22)|(textz>0)) & ~((landp>0)&(textz==0))).astype(np.uint8)
mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
np.save('M_halo.npy',mask)
# donors: clean ocean, eroded, excluding halo zones
donor=cv2.erode(oc,np.ones((9,9),np.uint8))&(a<0.02)
out=transplant(dh,mask,donor,valid_lp=cv2.erode(oc,np.ones((3,3),np.uint8))&(a<0.12))
cv2.imwrite('s1_nolabels.png',out)
