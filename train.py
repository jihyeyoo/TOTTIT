import torch.nn.functional as F
import colorsys

# ============================================================
# Dual Loss: Style별 다른 Loss 함수
# ============================================================
def rgb_to_hsv_batch(rgb):
    """
    RGB → HSV 변환 (배치)
    rgb: [N, 3] 텐서 (0~1 범위)
    return: [N, 3] (H, S, V)
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    max_c = torch.max(rgb, dim=1)[0]
    min_c = torch.min(rgb, dim=1)[0]
    delta = max_c - min_c
    
    # Value
    v = max_c
    
    # Saturation
    s = torch.where(max_c > 0, delta / max_c, torch.zeros_like(max_c))
    
    # Hue (간단 버전)
    h = torch.zeros_like(max_c)
    mask = (max_c == r) & (delta > 0)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
    mask = (max_c == g) & (delta > 0)
    h[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0
    mask = (max_c == b) & (delta > 0)
    h[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0
    h = h / 6.0
    
    return torch.stack([h, s, v], dim=1)


def calculate_palette_loss_dual(pred_images, gt_palettes, styles, 
                                 tone_on_tone_weight=0.05, 
                                 tone_in_tone_weight=0.1):
    """
    Style별 다른 Loss 전략
    
    Tone-on-Tone: Value-only Loss (약하게, 0.05)
    Tone-in-Tone: RGB Soft Assignment (강하게, 0.1)
    """
    images = (pred_images / 2 + 0.5).clamp(0, 1)
    targets = gt_palettes.float() / 255.0
    targets = targets.to(pred_images.device)

    small_images = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
    pixels = small_images.permute(0, 2, 3, 1).reshape(small_images.shape[0], -1, 3)

    total_loss = 0
    
    for i in range(pixels.shape[0]):
        p_rgb = pixels[i]  # [N, 3] RGB
        t_rgb = targets[i]  # [5, 3] RGB
        
        style = styles[i]
        
        # ============================================
        # Tone-on-Tone: Value-only Loss
        # ============================================
        if style == "tone_on_tone":
            # RGB → HSV 변환
            p_hsv = rgb_to_hsv_batch(p_rgb)  # [N, 3]
            t_hsv = rgb_to_hsv_batch(t_rgb)  # [5, 3]
            
            # Value만 추출
            p_v = p_hsv[:, 2].unsqueeze(1)  # [N, 1]
            t_v = t_hsv[:, 2].unsqueeze(0)  # [1, 5]
            
            # Value 거리 계산
            dist = torch.abs(p_v - t_v)  # [N, 5]
            
            # Hard Min (명도는 명확히 구분)
            min_dist, _ = torch.min(dist, dim=1)
            loss = torch.mean(min_dist) * tone_on_tone_weight
            
            total_loss += loss
        
        # ============================================
        # Tone-in-Tone: RGB Soft Assignment
        # ============================================
        else:  # tone_in_tone
            # RGB 거리 계산
            dist = torch.norm(p_rgb.unsqueeze(1) - t_rgb.unsqueeze(0), dim=2)
            
            # Soft Assignment (색 섞어도 OK)
            temperature = 0.5
            weights = F.softmax(-dist / temperature, dim=1)
            soft_dist = (dist * weights).sum(dim=1)
            
            loss = torch.mean(soft_dist) * tone_in_tone_weight
            
            total_loss += loss

    return total_loss / pixels.shape[0]


# ============================================================
# Dataset 
# ============================================================
class ToneAwareDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file, tokenizer, size=384, max_samples=None):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        if max_samples is not None:
            self.data = self.data[:max_samples]
            print(f"⚠️ Debug Mode: Using only {len(self.data)} samples!")
        
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item.get("image", item.get("image_path"))
        prompt = item["prompt"]
        gt_palette = torch.tensor(item["gt_palette"])
        style = item["style"] 

        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transforms(image)
        except:
            print(f"Warning: Failed to load {image_path}")
            pixel_values = torch.zeros(3, self.size, self.size)

        inputs = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, 
            padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
            "gt_palette": gt_palette,
            "style": style
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    gt_palettes = torch.stack([example["gt_palette"] for example in examples])
    styles = [example["style"] for example in examples]  # ★ List
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "gt_palette": gt_palettes,
        "styles": styles  
    }

# ============================================================
# 3. main
# ============================================================
def main():
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    DATA_FILE = "final_train_dataset.jsonl"
    OUTPUT_DIR = "tone_aware_lora_output"
 
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 2  # 2x2 = 4 (Effective Batch)
    LR = 2e-4
    NUM_EPOCHS = 8
    PALETTE_LOSS_WEIGHT = 0.1
    TEMPERATURE = 0.5  
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16" 
    )

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    # Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA setting
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=LR)

    dataset = ToneAwareDataset(DATA_FILE, tokenizer) 
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    lora_layers, optimizer, train_dataloader = accelerator.prepare(
        lora_layers, optimizer, train_dataloader
    )
    
    print(f"🚀 Training Started! Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE} (Accum: {GRADIENT_ACCUMULATION_STEPS})")
    print(f"📊 Soft Assignment Temperature: {TEMPERATURE}")

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        epoch_loss = 0.0
        epoch_diff = 0.0
        epoch_col = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Latent 생성
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Noise 추가
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. UNet 예측
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 4. Diffusion Loss
                loss_diff = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 5. Palette Loss 
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)
                alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                
                pred_original_sample = (
                    noisy_latents - torch.sqrt(1 - alpha_t) * model_pred
                ) / torch.sqrt(alpha_t)
                pred_original_sample = pred_original_sample.clamp(-1, 1)
                
                decoded_images = vae.decode(
                    pred_original_sample / vae.config.scaling_factor
                ).sample

                loss_color = calculate_palette_loss_dual(
                    decoded_images,
                    batch["gt_palette"],
                    batch["styles"], 
                    tone_on_tone_weight=0.05,  # 약하게
                    tone_in_tone_weight=0.10   # 강하게
                )

                loss = loss_diff + loss_color
                
                #loss = loss_diff + (PALETTE_LOSS_WEIGHT * loss_color)

                # 6. Backward
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # 로그 업데이트
            current_loss = loss.detach().item()
            current_diff = loss_diff.detach().item()
            current_col = loss_color.detach().item()
            
            epoch_loss += current_loss
            epoch_diff += current_diff
            epoch_col += current_col

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            logs = {
                "loss": f"{current_loss:.4f}", 
                "l_diff": f"{current_diff:.4f}", 
                "l_col": f"{current_col:.4f}"
            }
            progress_bar.set_postfix(**logs)
        
        # Epoch 종료 후 평균 출력
        avg_loss = epoch_loss / len(train_dataloader)
        avg_diff = epoch_diff / len(train_dataloader)
        avg_col = epoch_col / len(train_dataloader)
        print(f"\n✅ Epoch {epoch+1} Done! Avg Loss: {avg_loss:.4f} (Diff: {avg_diff:.4f}, Color: {avg_col:.4f})")
        
        # ★ [추가] Epoch마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-epoch{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unet.save_attn_procs(checkpoint_dir)
            print(f"💾 Checkpoint saved: {checkpoint_dir}")

    print("💾 Saving final LoRA weights...")
    unet.save_attn_procs(OUTPUT_DIR)
    print(f"🎉 Training Finished! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

