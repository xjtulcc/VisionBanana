import argparse
import gc

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLInstructPix2PixPipeline

from data.semantic_instruction_dataset import create_dataloader

MODEL_PATH = "models/sdxl-instructpix2pix"
MANIFEST_PATH = "datasets/semantic_instruction/train/manifest.jsonl"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_to_cpu(module):
    module.to("cpu")
    clear_memory()


def encode_prompt(pipe, prompts, device, dtype):
    """
    Encode prompts using two SDXL text encoders.
    """

    # Tokenizer 1
    tokens_1 = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Tokenizer 2
    tokens_2 = pipe.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Text Encoder 1
    pipe.text_encoder.to(device)

    with torch.no_grad():
        output_1 = pipe.text_encoder(
            tokens_1.input_ids.to(device),
            output_hidden_states=True,
        )

        hidden_1 = output_1.hidden_states[-2]

    move_to_cpu(pipe.text_encoder)

    # Text Encoder 2
    pipe.text_encoder_2.to(device)

    with torch.no_grad():
        output_2 = pipe.text_encoder_2(
            tokens_2.input_ids.to(device),
            output_hidden_states=True,
        )

        hidden_2 = output_2.hidden_states[-2]

        # Pooled embedding from encoder 2.
        pooled = output_2[0]

    move_to_cpu(pipe.text_encoder_2)

    # Concatenate token-level embeddings.
    prompt_embeds = torch.cat(
        [hidden_1, hidden_2],
        dim=-1,
    )

    return (
        prompt_embeds.to(device=device, dtype=dtype),
        pooled.to(device=device, dtype=dtype),
    )


def encode_images(pipe, images, device, dtype):
    """
    Encode images using VAE.
    """

    pipe.vae.to(device)

    images = images.to(
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        latents = pipe.vae.encode(images).latent_dist.sample()

        latents = latents * pipe.vae.config.scaling_factor

    move_to_cpu(pipe.vae)

    return latents


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[256, 512],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    dtype = torch.float16

    print("\n========== Environment ==========")

    print("PyTorch:", torch.__version__)

    print("CUDA:", torch.version.cuda)

    print("GPU:", torch.cuda.get_device_name(0))

    print("Resolution:", args.resolution)

    # --------------------------------------
    # 1. Load Pipeline
    # --------------------------------------

    print("\n========== Loading SDXL Pipeline ==========")

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        local_files_only=True,
    )

    pipe.set_progress_bar_config(disable=True)

    print("Pipeline loaded.")

    print("UNet input channels:", pipe.unet.config.in_channels)

    print("Cross attention dim:", pipe.unet.config.cross_attention_dim)

    print("VAE scaling factor:", pipe.vae.config.scaling_factor)

    print("Prediction type:", pipe.scheduler.config.prediction_type)

    # --------------------------------------
    # 2. Load Dataset
    # --------------------------------------

    print("\n========== Loading Dataset ==========")

    dataloader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    source = batch["conditioning_pixel_values"]

    target = batch["pixel_values"]

    prompts = batch["prompt"]

    print("Sample ID:", batch["sample_id"])

    print("Source:", source.shape)

    print("Target:", target.shape)

    print("Prompt:", prompts[0])

    # --------------------------------------
    # 3. Resize for optional VRAM reduction
    # --------------------------------------

    if args.resolution == 256:
        size = (256, 512)

        source = F.interpolate(
            source,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        target = F.interpolate(
            target,
            size=size,
            mode="nearest",
        )

    batch_size = source.shape[0]

    height = source.shape[2]

    width = source.shape[3]

    # --------------------------------------
    # 4. Text Encoding
    # --------------------------------------

    print("\n========== Text Encoding ==========")

    prompt_embeds, pooled_embeds = encode_prompt(
        pipe,
        prompts,
        device,
        dtype,
    )

    print("Prompt Embeddings:", prompt_embeds.shape)

    print("Pooled Embeddings:", pooled_embeds.shape)

    assert prompt_embeds.shape == (
        batch_size,
        77,
        2048,
    )

    # --------------------------------------
    # 5. VAE Encoding
    # --------------------------------------

    print("\n========== VAE Encoding ==========")

    source_latents = encode_images(
        pipe,
        source,
        device,
        dtype,
    )

    target_latents = encode_images(
        pipe,
        target,
        device,
        dtype,
    )

    print("Source Latents:", source_latents.shape)

    print("Target Latents:", target_latents.shape)

    # --------------------------------------
    # 6. Noise Generation
    # --------------------------------------

    print("\n========== Diffusion Noise ==========")

    noise = torch.randn_like(target_latents)

    timesteps = torch.randint(
        0,
        pipe.scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )

    noisy_target_latents = pipe.scheduler.add_noise(
        target_latents,
        noise,
        timesteps,
    )

    print("Noise:", noise.shape)

    print("Timesteps:", timesteps)

    print("Noisy Target:", noisy_target_latents.shape)

    # --------------------------------------
    # 7. UNet Input Construction
    # --------------------------------------

    print("\n========== UNet Input ==========")

    unet_input = torch.cat(
        [
            noisy_target_latents,
            source_latents,
        ],
        dim=1,
    )

    print("UNet Input:", unet_input.shape)

    assert unet_input.shape[1] == 8

    # --------------------------------------
    # 8. SDXL Time IDs
    # --------------------------------------

    original_size = (height, width)

    crops_coords_top_left = (0, 0)

    target_size = (height, width)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    add_time_ids = torch.tensor(
        [add_time_ids],
        device=device,
        dtype=dtype,
    )

    print("Time IDs:", add_time_ids)

    # --------------------------------------
    # 9. UNet Forward
    # --------------------------------------

    print("\n========== UNet Forward ==========")

    pipe.unet.to(device)

    added_cond_kwargs = {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
    }

    with torch.no_grad():
        noise_pred = pipe.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=True,
        ).sample

    print("Noise Prediction:", noise_pred.shape)

    assert noise_pred.shape == target_latents.shape

    # --------------------------------------
    # 10. Training Target
    # --------------------------------------

    prediction_type = pipe.scheduler.config.prediction_type

    if prediction_type == "epsilon":
        training_target = noise

    elif prediction_type == "v_prediction":
        training_target = pipe.scheduler.get_velocity(
            target_latents,
            noise,
            timesteps,
        )

    elif prediction_type == "sample":
        training_target = target_latents

    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    print("Prediction Type:", prediction_type)

    print("Training Target:", training_target.shape)

    # --------------------------------------
    # 11. Loss Calculation (No Backward)
    # --------------------------------------

    with torch.no_grad():
        loss = F.mse_loss(
            noise_pred.float(),
            training_target.float(),
        )

    print("Diffusion Loss:", loss.item())

    # --------------------------------------
    # 12. Final Validation
    # --------------------------------------

    assert torch.isfinite(noise_pred).all()

    assert torch.isfinite(training_target).all()

    assert torch.isfinite(loss)

    print("\n========== Validation Results ==========")

    print("Source Latents:", source_latents.shape)

    print("Target Latents:", target_latents.shape)

    print("Prompt Embeddings:", prompt_embeds.shape)

    print("Pooled Embeddings:", pooled_embeds.shape)

    print("UNet Input:", unet_input.shape)

    print("UNet Output:", noise_pred.shape)

    print("Loss:", loss.item())

    print("\nSDXL Training Input Test: PASSED")

    print("No backward propagation performed.")

    print("No model parameters updated.")

    move_to_cpu(pipe.unet)

    del pipe

    clear_memory()


if __name__ == "__main__":
    main()
