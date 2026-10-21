import json,subprocess,cv2,numpy as np,sys,unicodedata
im=cv2.imread(sys.argv[1] if len(sys.argv)>1 else 'final.png'); boxes=json.load(open('label_boxes.json'))
exp=['TERRAS DE LUMENOS','NORDGARD','TERRA SANTA','O REINO DE AEGIS','IMPÉRIO DE NERASHI']
ok_all=True
SC=float(sys.argv[2]) if len(sys.argv)>2 else 3
for (x0,y0,x1,y1),e in zip(boxes,exp):
    c=im[y0-10:y1+10,x0-14:x1+14].astype(int)
    ink=np.array([20,34,52])
    d=np.abs(c-ink).sum(2)
    b=np.where(d<90,0,255).astype(np.uint8)
    b=cv2.resize(b,None,fx=SC,fy=SC,interpolation=cv2.INTER_CUBIC); b=np.where(cv2.GaussianBlur(b,(0,0),1.2)>128,255,0).astype(np.uint8)
    b=cv2.copyMakeBorder(b,30,30,30,30,cv2.BORDER_CONSTANT,value=255)
    cv2.imwrite('crops/ocr.png',b)
    t=subprocess.run(['tesseract','crops/ocr.png','-','-l','por','--psm','7'],capture_output=True,text=True).stdout.strip()
    import re
    def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
    tt=re.sub(r'[^\wÀ-ÿ]','',unicodedata.normalize('NFC',t.upper())); ee=e.replace(' ','')
    ok=strip(tt)==strip(ee) and all(ch in tt for ch in ee if strip(ch)!=ch)  # expected accents must be present
    ok_all&=ok
    print(f'{e:20} OCR-> {t!r:26} {"OK" if ok else "DIFF"}')
print('ALL OK' if ok_all else 'FAIL')
