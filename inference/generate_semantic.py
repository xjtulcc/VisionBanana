import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file

from data.semantic_instruction_dataset import (
    SemanticInstructionDataset,
)

from training.train_semantic_lora import (
    MODEL_PATH,
    DEVICE,
    WEIGHT_DTYPE,
    load_encoders,
    encode_prompts,
    build_time_ids,
)


# ==========================================
# 1. Configuration
# ==========================================

DEFAULT_CHECKPOINT = "checkpoints/semantic_lora_smoke/checkpoint-2"

DEFAULT_MANIFEST = "datasets/semantic_instruction/val/manifest.jsonl"


# ==========================================
# 2. Load Checkpoint Metadata
# ==========================================


def load_checkpoint_metadata(checkpoint_dir):

    metadata_path = checkpoint_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_size = metadata.get("image_size", [512, 256])

    if isinstance(image_size, str):
        image_size = tuple(map(int, image_size.lower().split("x")))
    else:
        image_size = tuple(image_size)

    return metadata, image_size


# ==========================================
# 3. Load UNet + Trained LoRA
# ==========================================


def load_unet_with_lora(checkpoint_dir):

    print("\n========== Loading Original UNet ==========")

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    unet.to(DEVICE)

    # Freeze original parameters.
    unet.requires_grad_(False)

    print("\n========== Loading LoRA Configuration ==========")

    config = LoraConfig.from_pretrained(checkpoint_dir)

    unet.add_adapter(
        config,
        adapter_name="default",
    )

    # Match training dtype.
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    print("\n========== Loading Trained LoRA Weights ==========")

    state_dict = load_file(
        str(checkpoint_dir / "adapter_model.safetensors"),
        device="cpu",
    )

    result = set_peft_model_state_dict(
        unet,
        state_dict,
        adapter_name="default",
    )

    print("LoRA weights loaded.")

    print("Trainable adapter parameters:")

    trainable_params = [
        (name, param) for name, param in unet.named_parameters() if param.requires_grad
    ]

    print("Number of trainable tensors:", len(trainable_params))

    if not trainable_params:
        raise RuntimeError("No LoRA parameters were loaded.")

    # Verify the adapter has been loaded.
    for name, param in trainable_params:
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Invalid LoRA parameter: {name}")

    unet.eval()

    print("UNet + LoRA: READY")

    return unet


# ==========================================
# 4. Load Scheduler
# ==========================================


def load_noise_scheduler():

    scheduler = DDPMScheduler.from_pretrained(
        MODEL_PATH,
        subfolder="scheduler",
        local_files_only=True,
    )

    print("\n========== Scheduler ==========")

    print("Prediction Type:", scheduler.config.prediction_type)

    print("Training Timesteps:", scheduler.config.num_train_timesteps)

    return scheduler


# ==========================================
# 5. Encode Source Image
# ==========================================


@torch.no_grad()
def encode_source_image(source, vae):

    source = source.to(
        device="cpu",
        dtype=torch.float32,
    )

    source_latents = vae.encode(source).latent_dist.sample()

    source_latents = source_latents * vae.config.scaling_factor

    return source_latents.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )


# ==========================================
# 6. Decode Generated Latent
# ==========================================


@torch.no_grad()
def decode_generated_latent(latents, vae):

    latents = latents.to(
        device="cpu",
        dtype=torch.float32,
    )

    latents = latents / vae.config.scaling_factor

    decoded = vae.decode(latents).sample

    # [-1, 1] -> [0, 255]
    image = (decoded / 2 + 0.5).clamp(0, 1).cpu().numpy()

    image = image[0].transpose(1, 2, 0)

    image = (image * 255).round().astype(np.uint8)

    return Image.fromarray(image)


# ==========================================
# 7. Semantic Inference
# ==========================================


@torch.no_grad()
def generate_semantic(
    source,
    prompt,
    unet,
    vae,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    scheduler,
    image_size,
    num_inference_steps=30,
    seed=42,
):

    batch_size = source.shape[0]

    width, height = image_size

    # --------------------------------------
    # Text Encoding
    # --------------------------------------

    print("\n========== Text Encoding ==========")

    prompt_embeds, pooled_embeds = encode_prompts(
        [prompt],
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
    )

    # --------------------------------------
    # Source Encoding
    # --------------------------------------

    print("\n========== Source Encoding ==========")

    source_latents = encode_source_image(
        source,
        vae,
    )

    print("Source Latents:", source_latents.shape)

    # --------------------------------------
    # SDXL Additional Conditions
    # --------------------------------------

    time_ids = build_time_ids(
        batch_size,
        image_size,
    )

    added_cond_kwargs = {
        "text_embeds": pooled_embeds,
        "time_ids": time_ids,
    }

    # --------------------------------------
    # Initialize Random Latent
    # --------------------------------------

    print("\n========== Initialize Latent ==========")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    latent_height = height // 8
    latent_width = width // 8

    latents = torch.randn(
        (
            batch_size,
            4,
            latent_height,
            latent_width,
        ),
        generator=generator,
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    latents = latents * scheduler.init_noise_sigma

    print("Initial Latent:", latents.shape)

    # --------------------------------------
    # Set Inference Timesteps
    # --------------------------------------

    scheduler.set_timesteps(
        num_inference_steps,
        device=DEVICE,
    )

    print("\n========== Reverse Diffusion ==========")

    # --------------------------------------
    # Denoising Loop
    # --------------------------------------

    for index, timestep in enumerate(scheduler.timesteps):
        # Scale noisy target latent.
        latent_model_input = scheduler.scale_model_input(
            latents,
            timestep,
        )

        # InstructPix2Pix 8-channel input.
        unet_input = torch.cat(
            [
                latent_model_input,
                source_latents,
            ],
            dim=1,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            noise_pred = unet(
                unet_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True,
            ).sample

        # Reverse diffusion.
        latents = scheduler.step(
            noise_pred,
            timestep,
            latents,
            generator=generator,
        ).prev_sample

        if (
            index % max(1, num_inference_steps // 5) == 0
            or index == num_inference_steps - 1
        ):
            print(f"Step {index + 1}/{num_inference_steps}")

    print("Reverse Diffusion Completed.")

    # --------------------------------------
    # VAE Decoding
    # --------------------------------------

    print("\n========== VAE Decoding ==========")

    generated_image = decode_generated_latent(
        latents,
        vae,
    )

    print("Generated Image:", generated_image.size)

    return generated_image


# ==========================================
# 8. Main
# ==========================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        default="results/semantic_inference",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    checkpoint_dir = Path(args.checkpoint)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------
    # Environment
    # --------------------------------------

    print("\n========== Environment ==========")

    print("PyTorch:", torch.__version__)

    print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", checkpoint_dir)

    # --------------------------------------
    # Metadata
    # --------------------------------------

    metadata, image_size = load_checkpoint_metadata(checkpoint_dir)

    print("Image Size:", image_size)

    print("Training Step:", metadata.get("global_step"))

    # --------------------------------------
    # Dataset
    # --------------------------------------

    print("\n========== Loading Validation Dataset ==========")

    dataset = SemanticInstructionDataset(
        manifest_path=args.manifest,
        image_size=image_size,
    )

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError("Invalid sample_index.")

    sample = dataset[args.sample_index]

    source = sample["conditioning_pixel_values"].unsqueeze(0)

    prompt = sample["prompt"]

    sample_id = sample["sample_id"]

    print("Sample ID:", sample_id)

    print("Source Shape:", source.shape)

    print("Prompt:", prompt)

    # --------------------------------------
    # Frozen Encoders and VAE
    # --------------------------------------

    (
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        vae,
    ) = load_encoders()

    # --------------------------------------
    # Scheduler
    # --------------------------------------

    scheduler = load_noise_scheduler()

    # --------------------------------------
    # UNet + LoRA
    # --------------------------------------

    unet = load_unet_with_lora(checkpoint_dir)

    # --------------------------------------
    # Generate
    # --------------------------------------

    generated_image = generate_semantic(
        source=source,
        prompt=prompt,
        unet=unet,
        vae=vae,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        scheduler=scheduler,
        image_size=image_size,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    # --------------------------------------
    # Save Results
    # --------------------------------------

    source_image = (
        ((source[0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).round().astype(np.uint8)
    )

    source_image = Image.fromarray(source_image.transpose(1, 2, 0))

    source_path = output_dir / f"{sample_id}_source.png"

    generated_path = output_dir / f"{sample_id}_generated_rgb.png"

    source_image.save(source_path)

    generated_image.save(generated_path)

    with open(
        output_dir / f"{sample_id}_prompt.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(prompt)

    print("\n========== Results ==========")

    print("Source:", source_path.resolve())

    print("Generated RGB:", generated_path.resolve())

    print("\nSemantic Inference: COMPLETED")

    # --------------------------------------
    # Release Memory
    # --------------------------------------

    del unet
    del vae
    del text_encoder_1
    del text_encoder_2

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
