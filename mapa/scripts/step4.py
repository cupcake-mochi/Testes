import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
src='s3d.png'; im=cv2.imread(src); H,W=im.shape[:2]
rng=np.random.default_rng(7)
# ---------- parchment texture
def fbm(h,w,scales,amps):
    t=np.zeros((h,w),np.float32)
    for s,a in zip(scales,amps):
        n=rng.standard_normal((h//s+3,w//s+3)).astype(np.float32)
        n=cv2.resize(n,(w+3*s,h+3*s),interpolation=cv2.INTER_CUBIC)[:h,:w]
        t+=a*n
    return t
tex=fbm(H,W,[2,6,20,70],[3,4,6,8])
base=np.array([150,196,222],np.float32)  # BGR parchment
parch=base[None,None,:]+tex[...,None]*np.array([0.9,1.0,1.0])
# soft darkening toward the very outer edge (aged edge)
YY,XX=np.mgrid[:H,:W]
de=np.minimum(np.minimum(XX,W-1-XX),np.minimum(YY,H-1-YY)).astype(np.float32)
edge=np.clip(1-de/14,0,1)**1.6
parch=parch*(1-0.28*edge[...,None])
parch=np.clip(parch,0,255)
M=24  # margin
inner=(de>=M).astype(np.float32)
# inner shadow on the map near the neatline
sh=np.clip(1-(de-M)/10,0,1)*inner
mapimg=im.astype(np.float32)*(1-0.30*sh[...,None])
out=mapimg*inner[...,None]+parch*(1-inner[...,None])
out=np.clip(out,0,255).astype(np.uint8)
ink=(28,38,58) # BGR dark brown
def rect(img,inset,th,col):
    cv2.rectangle(img,(inset,inset),(W-1-inset,H-1-inset),col,th,lineType=cv2.LINE_AA)
rect(out,M-1,2,ink)       # main neatline (touching map)
rect(out,M-6,1,ink)       # thin double line
rect(out,5,1,(60,80,105)) # faint outer rule
# small corner squares
for (cx,cy) in [(M-3,M-3),(W-M+2,M-3),(M-3,H-M+2),(W-M+2,H-M+2)]:
    cv2.rectangle(out,(cx-6,cy-6),(cx+6,cy+6),ink,1,lineType=cv2.LINE_AA)
    cv2.rectangle(out,(cx-2,cy-2),(cx+2,cy+2),ink,-1,lineType=cv2.LINE_AA)
cv2.imwrite('s4_frame.png',out)
# ---------- labels
pil=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)).convert('RGBA')
F='fonts/IMFeENsc28P.ttf'
labels=[('Terras de Lumenos',(390,93),36),('Nordgard',(1085,56),36),('Terra Santa',(764,449),29),
        ('O Reino de Aegis',(256,532),36),('Império de Nerashi',(1112,630),36)]
inkc=(52,34,20,255); halo=(236,222,190,215)
lay=Image.new('RGBA',pil.size,(0,0,0,0)); d=ImageDraw.Draw(lay)
def draw_spaced(d,xy,text,font,fill,stroke=0,sf=None,track=2):
    x,y=xy
    for ch in text:
        d.text((x,y),ch,font=font,fill=fill,stroke_width=stroke,stroke_fill=sf)
        x+=font.getlength(ch)+track
def width(text,font,track=2): return sum(font.getlength(c)+track for c in text)-track
boxes=[]
for text,(cx,cy),sz in labels:
    font=ImageFont.truetype(F,sz)
    w=width(text,font); asc,desc=font.getmetrics()
    x=cx-w/2; y=cy-(asc+desc)/2-2
    boxes.append((int(x),int(y),int(x+w),int(y+asc+desc)))
    # soft tight halo (blurred stroke) then crisp stroke then ink
    hl=Image.new('RGBA',pil.size,(0,0,0,0)); hd=ImageDraw.Draw(hl)
    draw_spaced(hd,(x,y),text,font,(236,222,190,140),stroke=5,sf=(236,222,190,140))
    hl=hl.filter(ImageFilter.GaussianBlur(2.2))
    lay=Image.alpha_composite(lay,hl); d=ImageDraw.Draw(lay)
    draw_spaced(d,(x,y),text,font,inkc,stroke=2,sf=halo)
    draw_spaced(d,(x,y),text,font,inkc)
res=Image.alpha_composite(pil,lay).convert('RGB')
res.save('final.png')
import json; json.dump(boxes,open('label_boxes.json','w'))
print(boxes)
