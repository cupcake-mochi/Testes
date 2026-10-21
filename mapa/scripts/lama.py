import numpy as np, cv2, onnxruntime as ort
_s=None
def sess():
    global _s
    if _s is None:
        o=ort.SessionOptions(); o.intra_op_num_threads=4
        _s=ort.InferenceSession('models/lama.onnx',o)
    return _s
def run512(img,mask):
    x=img[...,::-1].astype(np.float32).transpose(2,0,1)[None]/255.
    m=(mask>0).astype(np.float32)[None,None]
    y=sess().run(None,{'image':x,'mask':m})[0][0].transpose(1,2,0)
    if y.max()<=1.5: y=y*255
    return np.clip(y,0,255)[...,::-1].astype(np.uint8)
def inpaint_region(img,mask,box,feather=3):
    """box=(x0,y0,x1,y1) window; inpaint mask inside window, scale to 512."""
    x0,y0,x1,y1=box
    H,W=img.shape[:2]; x0=max(0,x0);y0=max(0,y0);x1=min(W,x1);y1=min(H,y1)
    c=img[y0:y1,x0:x1]; m=mask[y0:y1,x0:x1]
    if m.max()==0: return img
    h,w=c.shape[:2]
    cs=cv2.resize(c,(512,512),interpolation=cv2.INTER_AREA if max(h,w)>512 else cv2.INTER_CUBIC)
    ms=cv2.resize(m,(512,512),interpolation=cv2.INTER_NEAREST)
    ms=cv2.dilate(ms,np.ones((3,3),np.uint8))
    out=run512(cs,ms)
    out=cv2.resize(out,(w,h),interpolation=cv2.INTER_CUBIC if max(h,w)>512 else cv2.INTER_AREA)
    a=(m>0).astype(np.float32)
    if feather: a=np.maximum(a,cv2.GaussianBlur(cv2.dilate(a,np.ones((feather*2+1,)*2,np.uint8)),(0,0),feather)*0.999)
    a=a[...,None]
    img=img.copy(); img[y0:y1,x0:x1]=(out*a+c*(1-a)).astype(np.uint8)
    return img
def inpaint_components(img,mask,ctx=2.2,minwin=200,maxwin=512):
    n,lab,st,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8))
    for i in range(1,n):
        x,y,w,h,a=st[i]
        cx,cy=x+w/2,y+h/2; s=int(max(minwin,min(maxwin,max(w,h)*ctx)))
        s=max(s,int(max(w,h)+60))
        img=inpaint_region(img,(lab==i).astype(np.uint8),(int(cx-s/2),int(cy-s/2),int(cx+s/2),int(cy+s/2)))
    return img
