import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

img = cv2.imread('ishihara.jpeg')

img_blur = cv2.GaussianBlur(img, (5,5), 0)

plt.figure(figsize=(6,6))
plt.title("Imagen con Gaussian Blur")
plt.imshow(img_blur, cmap="gray")
plt.axis("off")
plt.show()


hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

mask_g = cv2.inRange(hsv, np.array([25, 25, 25], np.uint8),
                          np.array([75,255,255], np.uint8))
mask_c = cv2.inRange(hsv, np.array([76, 15, 15], np.uint8),
                          np.array([105,255,255], np.uint8))
mask_num = cv2.bitwise_or(mask_g, mask_c)

plt.figure(figsize=(6,6))
plt.title("Máscara compuesta (verde ∪ cian)")
plt.imshow(mask_num, cmap="gray")
plt.axis("off")
plt.show()

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask_num = cv2.morphologyEx(mask_num, cv2.MORPH_OPEN,  k3, iterations=1)
solid    = cv2.morphologyEx(mask_num, cv2.MORPH_CLOSE, k7, iterations=2)
solid    = cv2.dilate(solid, np.ones((3,3), np.uint8), iterations=1)

plt.figure(figsize=(6,6))
plt.title("Después de suavizado")
plt.imshow(solid, cmap="gray")
plt.axis("off")
plt.show()

num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(solid, 8)
idxs = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > int(0.0015*solid.size)]

if len(idxs) == 0:
    print("Sin señal tras máscara."); masks_digits = [solid, solid]
else:
    xs = np.array([centroids[i,0] for i in idxs], dtype=np.float32).reshape(-1,1)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    # Si solo hay 1 comp, K=2 sobre los píxeles en X; si hay >=2 comps, K=2 sobre centroids
    if len(idxs) >= 2:
        _, lbls, centers = cv2.kmeans(xs, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        groups = {0: np.zeros_like(solid), 1: np.zeros_like(solid)}
        for lab, comp_i in zip(lbls.ravel(), idxs):
            m = (labels_cc == comp_i).astype(np.uint8)*255
            groups[lab] = cv2.bitwise_or(groups[lab], m)
        # Asegura orden izquierda→derecha
        left_id = int(np.argmin(centers))
        right_id = 1-left_id
        masks_digits = [groups[left_id], groups[right_id]]
    else:
        ys, xs_pix = np.where(solid > 0)
        data = xs_pix.reshape(-1,1).astype(np.float32)
        _, lbls, centers = cv2.kmeans(data, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        left_id = np.argmin(centers)
        m1 = np.zeros_like(solid); m2 = np.zeros_like(solid)
        sel = (lbls.ravel()==left_id)
        m1[ys[sel], xs_pix[sel]]   = 255
        m2[ys[~sel], xs_pix[~sel]] = 255
        masks_digits = [m1, m2]

def boost_top_bar(mask_bin, top_ratio=0.30, k=9):
    ys, xs = np.where(mask_bin>0)
    if xs.size==0: return mask_bin
    y1,y2 = ys.min(), ys.max()
    h = y2 - y1 + 1
    cut = y1 + int(h*top_ratio)  # 30% superior
    top = mask_bin.copy()
    top[:,:] = 0
    top[y1:cut, :] = mask_bin[y1:cut, :]
    top = cv2.dilate(top, cv2.getStructuringElement(cv2.MORPH_RECT,(k,1)), 1)
    # reinyecta la parte superior reforzada sin engordar el resto
    out = cv2.bitwise_or(mask_bin, top)
    # limpieza leve
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3, 1)
    return out

masks_digits[0] = boost_top_bar(masks_digits[0], top_ratio=0.30, k=11)  # marca la barra del 7
masks_digits[1] = cv2.morphologyEx(masks_digits[1], cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

def ocr_digit(m):
    if (m>0).sum()==0: return ""
    ys, xs = np.where(m>0)
    y1,y2 = ys.min(), ys.max(); x1,x2 = xs.min(), xs.max()
    d = m[y1:y2+1, x1:x2+1]
    d = cv2.bitwise_not(d)  # negro sobre blanco
    d = cv2.copyMakeBorder(d, 8,8,8,8, cv2.BORDER_CONSTANT, value=255)
    d = cv2.resize(d, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    cfg = "--oem 3 --psm 10 --dpi 300 -c tessedit_char_whitelist=0123456789 classify_bln_numeric_mode=1"
    return pytesseract.image_to_string(d, config=cfg).strip()

left  = ocr_digit(masks_digits[0])
right = ocr_digit(masks_digits[1])
print("Número detectado:", f"{left}{right}")
