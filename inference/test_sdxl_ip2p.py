import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sdxl-instructpix2pix"
RESULT_DIR = ROOT / "results"


def prepare_image(image_path, max_size=768):
    image = Image.open(image_path).convert("RGB")

    # 保持原始宽高比，限制最大边长
    scale = min(max_size / max(image.size), 1.0)

    width = max(8, round(image.width * scale / 8) * 8)
    height = max(8, round(image.height * scale / 8) * 8)

    image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Input image path")

    parser.add_argument(
        "--prompt",
        default="Make the image look like a watercolor painting.",
        help="Image editing instruction",
    )

    parser.add_argument("--steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1. 检查环境
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_DIR)
    print("Loading local model...")

    # 2. 加载本地模型
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )

    # 3. 显存优化
    pipe.enable_model_cpu_offload()

    # VAE slicing：如果当前版本支持，则启用
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    else:
        print("VAE slicing unavailable, skipping.")

    # 4. 准备输入图像
    image = prepare_image(image_path)

    print("Input size:", image.size)
    print("Prompt:", args.prompt)
    print("Starting inference...")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # 5. 执行图像生成
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            image=image,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            generator=generator,
        )

    output = result.images[0]

    # 6. 保存结果
    output_path = RESULT_DIR / "sdxl_ip2p_result.png"
    output.save(output_path)

    print("\nInference completed!")
    print("Output:", output_path)
    print("Output size:", output.size)

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
