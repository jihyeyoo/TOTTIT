import os
import json
import random
import colorsys
from PIL import Image, ImageDraw
from tqdm import tqdm

def generate_ideal_palette(style, label, n=5):
    palette = []

    if style == "tone_on_tone":
        # ===========================
        # 1. Hue 범위 (현실적 확장)
        # ===========================
        hue_map = {
            "beige": (0.08, 0.14),
            "grey":  (0.0, 1.0),
            "blue":  (0.53, 0.65),
            "green": (0.27, 0.40),
            "red":   (0.96, 0.04),  # wrap-around
        }

        if label == "red":
            base_h = random.uniform(0.96, 1.0) if random.random() < 0.7 else random.uniform(0.0, 0.04)
        else:
            base_h = random.uniform(*hue_map[label])

        # ===========================
        # 2. Saturation 범위 (현실 인테리어)
        # ===========================
        sat_map = {
            "beige": (0.18, 0.45),
            "grey":  (0.02, 0.10),
            "blue":  (0.15, 0.45),
            "green": (0.18, 0.55),
            "red":   (0.30, 0.60),
        }
        s_min, s_max = sat_map[label]

        # ===========================
        # 3. Value 범위 (현실 인테리어)
        # ===========================
        val_map = {
            "beige": (0.55, 0.95),
            "grey":  (0.35, 0.85),
            "blue":  (0.35, 0.85),
            "green": (0.35, 0.90),
            "red":   (0.40, 0.85),
        }
        v_min, v_max = val_map[label]

        # ===========================
        # 4. 팔레트 생성 (★ 핵심 개선!)
        # ===========================
        # (A) Hue variation
        h_var_map = {
            "blue": 0.05,
            "green": 0.08,
            "red": 0.04,
            "beige": 0.02,
            "grey": 1.0,
        }
        h_var = h_var_map[label]
        
        # (B) 명도 앵커 생성 (순서 유지 + 자연스러움)
        v_anchors = []
        v_step = (v_max - v_min) / (n - 1)
        
        for i in range(n):
            # 기본 앵커 (균등 분할)
            v_anchor = v_min + i * v_step
            
            # ★ 랜덤성 추가 (하지만 범위 제한)
            # 앞뒤 앵커와 겹치지 않도록
            noise_range = v_step * 0.3  # 30% 노이즈
            v_anchor += random.uniform(-noise_range, noise_range)
            
            # 범위 제한
            v_anchor = max(v_min, min(v_max, v_anchor))
            v_anchors.append(v_anchor)
        
        # (C) 앵커 정렬 (순서 보장)
        v_anchors.sort()
        
        # (D) 각 색상 생성
        for i in range(n):
            # Hue
            if label == "grey":
                h = random.random()
            else:
                h = (base_h + random.uniform(-h_var, h_var)) % 1.0
            
            # Saturation
            s = random.uniform(s_min, s_max)
            s += random.uniform(-0.05, 0.05)
            s = max(0, min(1, s))
            
            # Value (앵커 기반)
            v = v_anchors[i]
            v += random.uniform(-0.03, 0.03)  # 미세 조정
            v = max(v_min, min(v_max, v))
            
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette.append([int(r*255), int(g*255), int(b*255)])

        return palette

    # =========================================================
    # Tone-in-Tone (유지)
    # =========================================================
    elif style == "tone_in_tone":
        if label == "pastel":
            s_range, v_range = (0.20, 0.40), (0.80, 0.95)
        elif label == "vivid":
            s_range, v_range = (0.50, 0.75), (0.65, 0.90)
        elif label == "muted":
            s_range, v_range = (0.20, 0.50), (0.40, 0.70)
        elif label == "dark":
            s_range, v_range = (0.20, 0.60), (0.20, 0.45)
        elif label == "light":
            s_range, v_range = (0.10, 0.30), (0.85, 0.98)
        else:
            s_range, v_range = (0.30, 0.70), (0.40, 0.85)

        current_s = random.uniform(*s_range)
        current_v = random.uniform(*v_range)
        start_h = random.random()

        for i in range(n):
            h = (start_h + (i / n) + random.uniform(-0.02, 0.02)) % 1.0
            s = max(0, min(1, current_s + random.uniform(-0.05, 0.05)))
            
            if i == 0:
                v = random.uniform(0.15, 0.40)
            else:
                v = max(0, min(1, current_v + random.uniform(-0.05, 0.05)))
            
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette.append([int(r*255), int(g*255), int(b*255)])

        return palette



# ===========================================
# 설정 및 메인 로직 (동일)
# ===========================================
ROOT = "Pinterest"
OUT_JSONL = "final_train_dataset2.jsonl"
OUT_PALETTE_DIR = "gt_palette2"

def save_palette_image(palette, save_path, width=500, height=100):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    block_w = width // len(palette)
    for i, (r, g, b) in enumerate(palette):
        draw.rectangle([i * block_w, 0, (i + 1) * block_w, height], fill=(r, g, b))
    img.save(save_path)

def build_dataset():
    os.makedirs(OUT_PALETTE_DIR, exist_ok=True)
    
    f = open(OUT_JSONL, "w", encoding="utf-8")
    idx = 0

    for style in ["tone_on_tone", "tone_in_tone"]:
        style_dir = os.path.join(ROOT, style)
        if not os.path.exists(style_dir): continue

        for label in os.listdir(style_dir):
            label_dir = os.path.join(style_dir, label)
            if not os.path.isdir(label_dir): continue
            
            print(f"🚀 Processing {style} - {label} ...")
            
            for img_name in tqdm(os.listdir(label_dir)):
                if not img_name.lower().endswith(("jpg","jpeg","png","webp")):
                    continue
                
                img_path = os.path.abspath(os.path.join(label_dir, img_name))
                ideal_palette = generate_ideal_palette(style, label)
                
                pal_viz_path = os.path.join(OUT_PALETTE_DIR, f"{idx:05d}_{label}.png")
                save_palette_image(ideal_palette, pal_viz_path)
                
                item = {
                    "id": idx,
                    "image_path": img_path,
                    "style": style,
                    "label": label,
                    "gt_palette": ideal_palette,
                    "prompt": f"A {label} {style.replace('_','-')} interior design"
                }
                
                f.write(json.dumps(item) + "\n")
                idx += 1

    f.close()
    print(f"\n🎉 Finished! Total {idx} images.")
    print(f"💾 Dataset: {OUT_JSONL}")

if __name__ == "__main__":
    build_dataset()
