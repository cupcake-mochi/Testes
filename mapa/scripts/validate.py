import sys; sys.path.insert(0,'scripts')
import numpy as np, cv2
from ocean import ocean_mask
ARTS=[(150,392,236,455),(512,408,618,468),(240,286,300,364),(560,528,636,584),(760,396,822,444),
      (842,550,898,592),(880,650,960,705),(1100,660,1162,704),(970,12,1006,34),(1326,36,1386,68),
      (1214,68,1244,88),(1250,88,1268,104),(1312,90,1340,108),(1420,88,1440,102),(100,145,130,175),(1056,88,1080,110)]
LABELS=[(218,68,562,118),(993,32,1178,80),(663,426,866,472),(108,506,404,558),(952,606,1272,654)]
def base(im,inner):
    H,W=im.shape[:2]
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); Hh,S,V=[hsv[...,i].astype(int) for i in range(3)]
    oc=ocean_mask(im); roi=np.zeros((H,W),np.uint8); roi[inner:H-inner,inner:W-inner]=1
    nonoc=cv2.morphologyEx((1-oc).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))&roi
    return hsv,Hh,S,V,oc,roi,nonoc
def analyze(path,inner=28,ref=None):
    im=cv2.imread(path); H,W=im.shape[:2]
    hsv,Hh,S,V,oc,roi,nonoc=base(im,inner)
    green=((Hh>30)&(Hh<85)&(S>90)&(V<190)).astype(np.uint8)
    gz=cv2.dilate(cv2.morphologyEx(green,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)),np.ones((41,41),np.uint8))
    rep={}
    parch=((Hh>=5)&(Hh<=32)&(S>=25)&(S<=170)&(V>=150)).astype(np.uint8)&roi&(1-gz)
    parch=cv2.morphologyEx(parch,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    rep['V1 pergaminho fora da moldura (px)']=int(parch.sum())
    rem=[b for b in ARTS if nonoc[b[1]:b[3],b[0]:b[2]].sum()>15]
    rep['V2a artefatos conhecidos restantes']=rem
    if ref is not None:
        r=cv2.imread(ref); *_,ocr,roir,nonr=base(r,inner)
        n,lab,st,cen=cv2.connectedComponentsWithStats(nonoc)
        refz=cv2.dilate(nonr,np.ones((9,9),np.uint8))
        new=[(int(cen[i][0]),int(cen[i][1]),int(st[i][4])) for i in range(1,n) if st[i][4]>20 and refz[lab==i].mean()<0.5]
        rep['V2b objetos novos introduzidos']=new
    # text: dark ink pixels inside old label boxes
    f=im.astype(int); mx=f.max(2); mn=f.min(2); dark=((mx<70)&(mx-mn<30))
    rep['V3a tinta nos rotulos antigos (px)']=[int(dark[y0:y1,x0:x1].sum()) for (x0,y0,x1,y1) in LABELS]
    # halo: brightness anomaly in open ocean (>35px from any non-ocean)
    L=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[...,0].astype(np.float32)
    dist=cv2.distanceTransform(oc&roi,cv2.DIST_L2,5)
    v=(dist>30).astype(np.float32)
    def lp(s): return cv2.GaussianBlur(L*v,(0,0),s)/(cv2.GaussianBlur(v,(0,0),s)+1e-6)
    an=(lp(10)-lp(70))*(cv2.GaussianBlur(v,(0,0),10)>0.6)
    hal=cv2.morphologyEx((an>12).astype(np.uint8),cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
    n2,l2,s2,c2=cv2.connectedComponentsWithStats(hal)
    rep['V3b manchas claras (halo) em mar aberto']=[(int(c2[i][0]),int(c2[i][1]),int(s2[i][4])) for i in range(1,n2) if s2[i][4]>300]
    # texture energy consistency in open ocean
    hp=L-cv2.GaussianBlur(L,(0,0),3); vv=(dist>8).astype(np.float32)
    e=cv2.GaussianBlur(hp*hp*vv,(0,0),12)/(cv2.GaussianBlur(vv,(0,0),12)+1e-6); ok=cv2.GaussianBlur(vv,(0,0),12)>0.7
    ratio=e/np.median(e[ok])
    low=cv2.morphologyEx(((ratio<0.4)&ok).astype(np.uint8),cv2.MORPH_OPEN,np.ones((15,15),np.uint8))
    hi=cv2.morphologyEx(((ratio>2.8)&ok).astype(np.uint8),cv2.MORPH_OPEN,np.ones((15,15),np.uint8))
    def blobs(m,t=500):
        n,l,s,c=cv2.connectedComponentsWithStats(m); return [(int(c[i][0]),int(c[i][1]),int(s[i][4])) for i in range(1,n) if s[i][4]>t]
    rep['V4 manchas borradas/ruidosas no mar']=blobs(low)+blobs(hi)
    # straight axis-aligned seams in open ocean tone
    l=cv2.GaussianBlur(L,(0,0),3)
    ed=(cv2.Canny(np.clip(l,0,255).astype(np.uint8),6,14)>0)&(dist>6)
    ed=ed.astype(np.uint8)
    hl=cv2.morphologyEx(ed,cv2.MORPH_OPEN,np.ones((1,45),np.uint8)); vl=cv2.morphologyEx(ed,cv2.MORPH_OPEN,np.ones((45,1),np.uint8))
    rep['V5 emendas retas (px)']=int(hl.sum()+vl.sum())
    ys,xs=np.nonzero(hl|vl); rep['V5 locais']=sorted(set((int(x)//40*40,int(y)//40*40) for x,y in zip(xs,ys)))[:20]
    return rep
if __name__=='__main__':
    ref=None
    for p in sys.argv[1:]:
        r=analyze(p,28,ref='orig.png' if p!='orig.png' else None)
        print('==',p)
        for k,v in r.items(): print('  ',k,':',v)
