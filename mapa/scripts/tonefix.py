import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from ocean import ocean_mask, tone_field, lowpass_masked
src,dst=sys.argv[1],sys.argv[2]
im=cv2.imread(src); H,W=im.shape[:2]
pts=[(155,401),(565,571),(610,599),(750,135),(1312,860),(69,714)]
YY,XX=np.mgrid[:H,:W]; wgt=np.zeros((H,W),np.float32)
for x,y in pts: wgt=np.maximum(wgt,np.clip((1-np.hypot(XX-x,YY-y)/70)/0.35,0,1))
oc=ocean_mask(im); dist=cv2.distanceTransform(oc,cv2.DIST_L2,5)
ocw=np.clip((dist-3)/8,0,1)
G=tone_field(im,((dist>12)&(wgt<0.02)).astype(np.uint8),sigmas=(30,60,120))
lp=lowpass_masked(im,(dist>4),10)
out=np.clip(im.astype(np.float32)+(G-lp)*(wgt*ocw)[...,None],0,255).astype(np.uint8)
cv2.imwrite(dst,out)
