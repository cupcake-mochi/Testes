import numpy as np, cv2
def ocean_mask(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); H,S,V=[hsv[...,i].astype(int) for i in range(3)]
    oc=((H>=80)&(H<=112)&(S>45)).astype(np.uint8)
    oc=cv2.morphologyEx(oc,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    oc=cv2.morphologyEx(oc,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    return oc
def lowpass_masked(img,valid,s):
    v=valid.astype(np.float32)
    num=cv2.GaussianBlur(img.astype(np.float32)*v[...,None],(0,0),s)
    den=cv2.GaussianBlur(v,(0,0),s)[...,None]+1e-6
    return num/den
def transplant(img,mask,donor_ok,exclude=None,s=7,feather=7,rng=None,search=None,valid_lp=None,tile=(150,90),G=None):
    """Fill mask with real texture from a donor window fully inside donor_ok."""
    rng=rng or np.random.default_rng(0)
    H,W=img.shape[:2]
    n,lab0,st0,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8))
    # split big components into tiles
    lab=np.zeros_like(lab0); k=0; boxes=[]
    for i in range(1,n):
        x,y,w,h,_a=st0[i]
        nx=max(1,int(np.ceil(w/tile[0]))); ny=max(1,int(np.ceil(h/tile[1])))
        for ty in range(ny):
            for tx in range(nx):
                xa=x+tx*w//nx; xb=x+(tx+1)*w//nx; ya=y+ty*h//ny; yb=y+(ty+1)*h//ny
                sub=(lab0[ya:yb,xa:xb]==i)
                if sub.any():
                    k+=1; lab[ya:yb,xa:xb][sub]=k
    n=k+1
    st=np.zeros((n,5),int)
    for i in range(1,n):
        ys,xs=np.nonzero(lab==i); st[i]=[xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1,len(xs)]
    out=img.copy()
    f=img.astype(np.float32)
    for i in range(1,n):
        f=out.astype(np.float32)
        x,y,w,h,_a=st[i]; pad=feather*2+4
        x0,y0=max(0,x-pad),max(0,y-pad); x1,y1=min(W,x+w+pad),min(H,y+h+pad)
        bw,bh=x1-x0,y1-y0
        comp=(lab[y0:y1,x0:x1]==i)
        # valid context for target low-pass: outside all masks
        valid=(lab==0)|(lab<i)
        if exclude is not None: valid&=(exclude==0)
        if valid_lp is not None: valid&=(valid_lp>0)
        # target lowpass (bigger window for context)
        cx0,cy0=max(0,x0-120),max(0,y0-120); cx1,cy1=min(W,x1+120),min(H,y1+120)
        if G is not None: lp=G[y0:y1,x0:x1]
        else:
            lp=lowpass_masked(f[cy0:cy1,cx0:cx1],valid[cy0:cy1,cx0:cx1],5*s)
            lp=lp[y0-cy0:y0-cy0+bh,x0-cx0:x0-cx0+bw]
        # find donor windows
        ok=(donor_ok>0).astype(np.uint8); ii=cv2.integral(ok).astype(np.int64)
        tot=ii[bh:,bw:]-ii[:-bh,bw:]-ii[bh:,:-bw]+ii[:-bh,:-bw]
        cand=np.argwhere(tot==bw*bh)
        if len(cand)==0: print('no donor for comp',i,(x,y,w,h)); continue
        tgt_mean=lp[comp].mean(0)
        d=np.hypot(cand[:,0]-y0,cand[:,1]-x0)
        # sample a subset, choose by colour+distance
        idx=rng.choice(len(cand),min(4000,len(cand)),replace=False)
        best=None;bs=1e18
        for j in idx:
            dy,dx=cand[j]; dpatch=f[dy:dy+bh,dx:dx+bw]
            m=cv2.GaussianBlur(dpatch,(0,0),3*s).reshape(-1,3).mean(0)
            sc=np.abs(m-tgt_mean).sum()+0.02*d[j]
            if sc<bs: bs=sc;best=(dy,dx)
        dy,dx=best; dpatch=f[dy:dy+bh,dx:dx+bw]
        detail=dpatch-cv2.GaussianBlur(dpatch,(0,0),s)
        dlp=cv2.GaussianBlur(dpatch,(0,0),s)
        # match low-freq: use target lp, preserve donor's mid-tone variance slightly
        fill=lp+detail+0.0*(dlp-dlp.mean((0,1)))
        a=cv2.GaussianBlur(cv2.dilate(comp.astype(np.uint8),np.ones((feather,feather),np.uint8)).astype(np.float32),(0,0),feather/2)
        a=np.maximum(a,comp.astype(np.float32))[...,None]
        reg=out[y0:y1,x0:x1].astype(np.float32)
        out[y0:y1,x0:x1]=np.clip(fill*a+reg*(1-a),0,255).astype(np.uint8)
    return out

def tone_field(img,valid,sigmas=(25,50,100,200)):
    f=img.astype(np.float32); v=valid.astype(np.float32)
    G=None; have=np.zeros(v.shape,np.float32)
    for sg in sigmas:
        num=cv2.GaussianBlur(f*v[...,None],(0,0),sg); den=cv2.GaussianBlur(v,(0,0),sg)
        est=num/(den[...,None]+1e-6); conf=np.clip(den/0.15,0,1)
        if G is None: G=est; have=conf
        else:
            w=(1-have)[...,None]; G=G*(1-w)+est*w; have=np.maximum(have,conf)
    # final smoothing
    return cv2.GaussianBlur(G,(0,0),6)
