# === 回档到多点色差mask+自动切块检测优化版 ===
import cv2
import numpy as np
from PIL import Image
import os

img_path = r"C:/Users/Lenovo/Pictures/Status/characters/idle.png"  # 修改为你的png路径
out_dir = r"./tools/sprite_character_output"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(img_path).convert("RGBA")
img_np = np.array(img)
h, w = img_np.shape[0], img_np.shape[1]

# 自动检测切块区域（假定每帧间有明显空白/棋盘格）
# 先灰度化+二值化，连通域分析，自动找到所有非背景块
img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
# 多点背景色采样
bg_samples = [img_np[0,0,:3], img_np[0,-1,:3], img_np[-1,0,:3], img_np[-1,-1,:3], img_np[h//2,0,:3], img_np[0,w//2,:3], img_np[h//2,w//2,:3]]
bg_samples = np.array(bg_samples)
flat_img = img_np[...,:3].reshape(-1,3)
dist = np.min(np.linalg.norm(flat_img[:,None,:] - bg_samples[None,:,:], axis=2), axis=1)
# 用较低阈值，确保最大外轮廓能包裹全部角色
mask_fg = (dist > 60).astype(np.uint8).reshape(h, w) * 255
kernel = np.ones((5,5), np.uint8)
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
# mask_fg = cv2.erode(mask_fg, kernel, iterations=1)  # 已去除腐蚀操作
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

# 连通域提取所有角色帧区域
contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
max_extract = 12  # 最多提取帧数
idx = 1
for cnt in contours[:max_extract]:
    if cv2.contourArea(cnt) < 500:  # 过滤小块误切割
        continue
    x, y, w2, h2 = cv2.boundingRect(cnt)
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    w2 = min(img_np.shape[1] - x, w2 + 2 * pad)
    h2 = min(img_np.shape[0] - y, h2 + 2 * pad)
    roi = img_np[y:y+h2, x:x+w2].copy()
    # 只保留roi内前景mask
    roi_mask = mask_fg[y:y+h2, x:x+w2]
    roi[...,3] = roi_mask
    out_pil = Image.fromarray(roi)
    out_pil.save(os.path.join(out_dir, f"frame_{idx:02d}.png"))
    idx += 1

print(f"已输出{idx-1}帧角色透明图到 {out_dir}")