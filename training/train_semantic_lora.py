import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from data.semantic_instruction_dataset import (
    SemanticInstructionDataset,
)
from training.lora_config import (
    configure_unet_lora,
    print_trainable_parameters,
)

# ==========================================
# 1. Configuration
# ==========================================

MODEL_PATH = "models/sdxl-instructpix2pix"

TRAIN_MANIFEST = "datasets/semantic_instruction/train/manifest.jsonl"

VAL_MANIFEST = "datasets/semantic_instruction/val/manifest.jsonl"

DEVICE = torch.device("cuda")

WEIGHT_DTYPE = torch.float16


# ==========================================
# 2. Utilities
# ==========================================


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_memory():

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3

        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU Allocated: {allocated:.2f} GB")

        print(f"GPU Reserved: {reserved:.2f} GB")


def parse_resolution(value):

    try:
        width, height = map(
            int,
            value.lower().split("x"),
        )

        if width <= 0 or height <= 0:
            raise ValueError

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError

        return width, height

    except ValueError:
        raise argparse.ArgumentTypeError(
            "Resolution must be WIDTHxHEIGHT, with both dimensions divisible by 8."
        )


# ==========================================
# 3. Dataset
# ==========================================


def build_dataloader(
    manifest_path,
    image_size,
    batch_size,
    shuffle,
    num_workers,
):

    dataset = SemanticInstructionDataset(
        manifest_path=manifest_path,
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return dataset, dataloader


# ==========================================
# 4. Load Frozen Encoders
# ==========================================


def load_encoders():

    model_path = Path(MODEL_PATH)

    print("\n========== Loading Tokenizers ==========")

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer",
        local_files_only=True,
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer_2",
        local_files_only=True,
    )

    print("\n========== Loading Text Encoders ==========")

    # Keep text encoders on CPU to save GPU memory.
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_path / "text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path / "text_encoder_2",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    text_encoder_1.eval()
    text_encoder_2.eval()

    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    print("\n========== Loading VAE ==========")

    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    vae.eval()
    vae.requires_grad_(False)

    print("Text Encoders: CPU / FP32")
    print("VAE: CPU / FP32")

    return (
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        vae,
    )


# ==========================================
# 5. Encode Prompt
# ==========================================


@torch.no_grad()
def encode_prompts(
    prompts,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
):

    tokens_1 = tokenizer_1(
        prompts,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    tokens_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    output_1 = text_encoder_1(
        tokens_1.input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    output_2 = text_encoder_2(
        tokens_2.input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    prompt_embeds_1 = output_1.hidden_states[-2]

    prompt_embeds_2 = output_2.hidden_states[-2]

    prompt_embeds = torch.cat(
        [
            prompt_embeds_1,
            prompt_embeds_2,
        ],
        dim=-1,
    )

    pooled_embeds = output_2.text_embeds

    prompt_embeds = prompt_embeds.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    pooled_embeds = pooled_embeds.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    return prompt_embeds, pooled_embeds


# ==========================================
# 6. Encode Images
# ==========================================


@torch.no_grad()
def encode_images(
    source,
    target,
    vae,
):

    source = source.to(
        device="cpu",
        dtype=torch.float32,
    )

    target = target.to(
        device="cpu",
        dtype=torch.float32,
    )

    source_latents = vae.encode(source).latent_dist.sample()

    target_latents = vae.encode(target).latent_dist.sample()

    scaling_factor = vae.config.scaling_factor

    source_latents = source_latents * scaling_factor

    target_latents = target_latents * scaling_factor

    source_latents = source_latents.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    target_latents = target_latents.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    return source_latents, target_latents


# ==========================================
# 7. SDXL Time IDs
# ==========================================


def build_time_ids(
    batch_size,
    image_size,
):

    width, height = image_size

    values = [
        height,
        width,
        0,
        0,
        height,
        width,
    ]

    time_ids = torch.tensor(
        values,
        dtype=WEIGHT_DTYPE,
        device=DEVICE,
    )

    return time_ids.unsqueeze(0).repeat(
        batch_size,
        1,
    )


# ==========================================
# 8. Load UNet + LoRA
# ==========================================


def load_unet():

    print("\n========== Loading UNet ==========")

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    unet.to(DEVICE)

    # Freeze base UNet and inject LoRA.
    configure_unet_lora(unet)

    # Keep trainable adapters in FP32.
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Gradient checkpointing.
    unet.enable_gradient_checkpointing()

    # Verify only LoRA parameters are trainable.
    statistics = print_trainable_parameters(unet)

    return unet, statistics


# ==========================================
# 9. Diffusion Forward
# ==========================================


def diffusion_forward(
    batch,
    unet,
    vae,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    noise_scheduler,
    image_size,
):

    source = batch["conditioning_pixel_values"]

    target = batch["pixel_values"]

    prompts = batch["prompt"]

    batch_size = source.shape[0]

    # --------------------------------------
    # Text Encoding
    # --------------------------------------

    prompt_embeds, pooled_embeds = encode_prompts(
        prompts,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
    )

    # --------------------------------------
    # VAE Encoding
    # --------------------------------------

    source_latents, target_latents = encode_images(
        source,
        target,
        vae,
    )

    # --------------------------------------
    # Sample Noise and Timesteps
    # --------------------------------------

    noise = torch.randn_like(target_latents)

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=DEVICE,
    ).long()

    noisy_target = noise_scheduler.add_noise(
        target_latents,
        noise,
        timesteps,
    )

    # --------------------------------------
    # InstructPix2Pix 8-channel Input
    # --------------------------------------

    unet_input = torch.cat(
        [
            noisy_target,
            source_latents,
        ],
        dim=1,
    )

    # --------------------------------------
    # SDXL Conditioning
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
    # Prediction Target
    # --------------------------------------

    prediction_type = noise_scheduler.config.prediction_type

    if prediction_type == "epsilon":
        training_target = noise

    elif prediction_type == "v_prediction":
        training_target = noise_scheduler.get_velocity(
            target_latents,
            noise,
            timesteps,
        )

    elif prediction_type == "sample":
        training_target = target_latents

    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    # --------------------------------------
    # UNet Forward
    # --------------------------------------

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        noise_pred = unet(
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=True,
        ).sample

    # --------------------------------------
    # Diffusion Loss
    # --------------------------------------

    loss = F.mse_loss(
        noise_pred.float(),
        training_target.float(),
    )

    if not torch.isfinite(loss):
        raise RuntimeError("Loss contains NaN or Inf.")

    return loss


# ==========================================
# 10. Save Checkpoint
# ==========================================


def save_checkpoint(
    unet,
    optimizer,
    lr_scheduler,
    scaler,
    output_dir,
    epoch,
    global_step,
    loss,
    args,
):

    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save LoRA config.
    unet.peft_config["default"].save_pretrained(checkpoint_dir)

    # Save LoRA weights only.
    state_dict = get_peft_model_state_dict(
        unet,
        adapter_name="default",
    )

    state_dict = {
        key: value.detach().cpu().contiguous() for key, value in state_dict.items()
    }

    save_file(
        state_dict,
        str(checkpoint_dir / "adapter_model.safetensors"),
    )

    # Save optimizer and training progress separately.
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "loss": float(loss),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )

    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": float(loss),
        "learning_rate": args.learning_rate,
        "rank": 8,
        "image_size": args.resolution,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": (args.gradient_accumulation_steps),
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }

    with open(
        checkpoint_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nCheckpoint saved: {checkpoint_dir}")


# ==========================================
# 11. Resume Checkpoint
# ==========================================


def resume_checkpoint(
    unet,
    optimizer,
    lr_scheduler,
    scaler,
    checkpoint_dir,
):

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)

    print("\n========== Resuming Training ==========")

    config = LoraConfig.from_pretrained(checkpoint_dir)

    # Remove the initially injected adapter.
    unet.delete_adapters("default")

    # Inject checkpoint configuration.
    unet.add_adapter(
        config,
        adapter_name="default",
    )

    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    state_dict = load_file(
        str(checkpoint_dir / "adapter_model.safetensors"),
        device="cpu",
    )

    set_peft_model_state_dict(
        unet,
        state_dict,
        adapter_name="default",
    )

    state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    optimizer.load_state_dict(state["optimizer"])

    lr_scheduler.load_state_dict(state["lr_scheduler"])

    scaler.load_state_dict(state["scaler"])

    print("Resumed Epoch:", state["epoch"])

    print("Resumed Global Step:", state["global_step"])

    return (
        state["epoch"],
        state["global_step"],
    )


# ==========================================
# 12. Validation
# ==========================================


@torch.no_grad()
def validate(
    dataloader,
    unet,
    vae,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    noise_scheduler,
    image_size,
):

    print("\n========== Validation ==========")

    unet.eval()

    losses = []

    for batch in dataloader:
        loss = diffusion_forward(
            batch,
            unet,
            vae,
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            noise_scheduler,
            image_size,
        )

        losses.append(loss.item())

    validation_loss = float(np.mean(losses))

    print(f"Validation Loss: {validation_loss:.6f}")

    unet.train()

    return validation_loss


# ==========================================
# 13. Argument Parser
# ==========================================


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_manifest",
        default=TRAIN_MANIFEST,
    )

    parser.add_argument(
        "--val_manifest",
        default=VAL_MANIFEST,
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoints/semantic_lora",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(512, 256),
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ==========================================
# 14. Main Training
# ==========================================


def main():

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation must be positive.")

    if args.save_steps <= 0:
        raise ValueError("save_steps must be positive.")

    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    print("\n========== Environment ==========")

    print("PyTorch:", torch.__version__)

    print("GPU:", torch.cuda.get_device_name(0))

    print("Resolution:", args.resolution)

    print("Batch Size:", args.batch_size)

    print("Epochs:", args.epochs)

    print("Gradient Accumulation:", args.gradient_accumulation_steps)

    print_memory()

    # --------------------------------------
    # Dataset
    # --------------------------------------

    print("\n========== Loading Datasets ==========")

    train_dataset, train_loader = build_dataloader(
        args.train_manifest,
        args.resolution,
        args.batch_size,
        True,
        args.num_workers,
    )

    val_dataset, val_loader = build_dataloader(
        args.val_manifest,
        args.resolution,
        args.batch_size,
        False,
        args.num_workers,
    )

    print("Train Samples:", len(train_dataset))

    print("Validation Samples:", len(val_dataset))

    print(
        "\nWARNING: Small datasets are suitable for pipeline "
        "validation, not final performance claims."
    )

    # --------------------------------------
    # Frozen Encoders
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

    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_PATH,
        subfolder="scheduler",
        local_files_only=True,
    )

    print(
        "Prediction Type:",
        noise_scheduler.config.prediction_type,
    )

    # --------------------------------------
    # UNet + LoRA
    # --------------------------------------

    unet, statistics = load_unet()

    trainable_params = [param for param in unet.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=1e-2,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)

    total_steps = (
        args.max_train_steps
        if args.max_train_steps is not None
        else updates_per_epoch * args.epochs
    )

    if total_steps <= 0:
        raise ValueError("Total training steps must be positive.")

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(
                (step + 1) / max(args.warmup_steps, 1),
                1.0,
            )
            if step < args.warmup_steps
            else max(
                0.0,
                (total_steps - step) / max(total_steps - args.warmup_steps, 1),
            )
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        start_epoch, global_step = resume_checkpoint(
            unet,
            optimizer,
            lr_scheduler,
            scaler,
            args.resume_from_checkpoint,
        )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------
    # Training Loop
    # --------------------------------------

    print("\n========== Training Started ==========")

    stop_training = False

    for epoch in range(start_epoch, args.epochs):
        unet.train()

        epoch_losses = []

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for batch_index, batch in enumerate(train_loader):
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps

            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                loss = diffusion_forward(
                    batch,
                    unet,
                    vae,
                    tokenizer_1,
                    tokenizer_2,
                    text_encoder_1,
                    text_encoder_2,
                    noise_scheduler,
                    args.resolution,
                )

            epoch_losses.append(loss.item())

            scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            is_accumulation_boundary = (
                batch_index + 1
            ) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                train_loader
            )

            if not is_accumulation_boundary:
                continue

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                args.max_grad_norm,
            )

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step()

            global_step += 1

            print(
                f"Epoch: {epoch + 1} | "
                f"Step: {global_step}/{total_steps} | "
                f"Batch Loss: {loss.item():.6f} | "
                f"LR: {lr_scheduler.get_last_lr()[0]:.8f}"
            )

            if global_step % args.save_steps == 0:
                save_checkpoint(
                    unet,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    args.output_dir,
                    epoch + 1,
                    global_step,
                    loss.item(),
                    args,
                )

            if global_step >= total_steps:
                stop_training = True
                break

        print(f"Epoch Mean Loss: {np.mean(epoch_losses):.6f}")

        # ----------------------------------
        # Validation
        # ----------------------------------

        validation_loss = validate(
            val_loader,
            unet,
            vae,
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            noise_scheduler,
            args.resolution,
        )

        print(f"Epoch {epoch + 1} Validation Loss: {validation_loss:.6f}")

        if stop_training:
            break

    # --------------------------------------
    # Final Checkpoint
    # --------------------------------------

    print("\n========== Saving Final Checkpoint ==========")

    final_dir = Path(args.output_dir) / f"checkpoint-{global_step}"

    save_checkpoint(
        unet,
        optimizer,
        lr_scheduler,
        scaler,
        args.output_dir,
        epoch + 1,
        global_step,
        epoch_losses[-1],
        args,
    )

    print("\n========== Training Completed ==========")

    print("Global Steps:", global_step)

    print("Trainable Parameters:", statistics["trainable_params"])

    print("Final Checkpoint:", final_dir)

    print_memory()

    # Release resources.
    del optimizer
    del unet
    del vae
    del text_encoder_1
    del text_encoder_2

    clear_memory()


if __name__ == "__main__":
    main()
