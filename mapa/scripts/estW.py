import numpy as np,cv2
im=cv2.imread('orig.png'); T=np.load('T_text.npy')
est=cv2.inpaint(im,T,5,cv2.INPAINT_TELEA).astype(np.float32)
L=cv2.GaussianBlur(est,(0,0),6)
# sample deep core vs nearby ocean outside, per label
for k,(core,out) in {'lum':((380,95),(380,150)),'aeg':((250,532),(250,600)),'ner':((1110,630),(1110,580)),'nord':((1080,55),(1080,10)),'santa':((760,450),(700,500))}.items():
    c=L[core[1],core[0]]; o=L[out[1],out[0]]
    print(k,'core',c.round(),'out',o.round(),'diff ratio',((c-o)/(c-o).max()).round(2))
