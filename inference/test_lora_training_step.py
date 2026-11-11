import gc
from pathlib import Path

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
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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

MANIFEST_PATH = "datasets/semantic_instruction/train/manifest.jsonl"

# PIL size: (width, height)
IMAGE_SIZE = (512, 256)

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

DEVICE = "cuda"

WEIGHT_DTYPE = torch.float16


# ==========================================
# 2. Memory Management
# ==========================================


def clear_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_memory():

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3

        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU Allocated: {allocated:.2f} GB")

        print(f"GPU Reserved: {reserved:.2f} GB")


# ==========================================
# 3. Load Dataset
# ==========================================


def load_training_batch():

    dataset = SemanticInstructionDataset(
        manifest_path=MANIFEST_PATH,
        image_size=IMAGE_SIZE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    batch = next(iter(dataloader))

    source = batch["conditioning_pixel_values"]

    target = batch["pixel_values"]

    prompts = batch["prompt"]

    sample_ids = batch["sample_id"]

    print("\n========== Dataset ==========")

    print("Sample IDs:", sample_ids)

    print("Source:", source.shape)

    print("Target:", target.shape)

    print("Prompt:", prompts[0][:150], "...")

    assert source.shape == target.shape

    assert source.shape[0] == BATCH_SIZE

    return source, target, prompts


# ==========================================
# 4. Encode Text
# ==========================================


def encode_prompts(prompts):

    print("\n========== Text Encoding ==========")

    model_path = Path(MODEL_PATH)

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer",
        local_files_only=True,
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer_2",
        local_files_only=True,
    )

    # --------------------------------------
    # Text Encoder 1
    # --------------------------------------

    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_path / "text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    text_encoder_1.eval()

    tokens_1 = tokenizer_1(
        prompts,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_1 = text_encoder_1(
            tokens_1.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        prompt_embeds_1 = output_1.hidden_states[-2]

    del text_encoder_1, output_1, tokens_1

    clear_memory()

    # --------------------------------------
    # Text Encoder 2
    # --------------------------------------

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path / "text_encoder_2",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    text_encoder_2.eval()

    tokens_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_2 = text_encoder_2(
            tokens_2.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        prompt_embeds_2 = output_2.hidden_states[-2]

        pooled_prompt_embeds = output_2.text_embeds

    # Concatenate the two text encoders.
    prompt_embeds = torch.cat(
        [prompt_embeds_1, prompt_embeds_2],
        dim=-1,
    )

    # Move embeddings to GPU.
    prompt_embeds = prompt_embeds.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    pooled_prompt_embeds = pooled_prompt_embeds.to(
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
    )

    print("Prompt Embeddings:", prompt_embeds.shape)

    print("Pooled Embeddings:", pooled_prompt_embeds.shape)

    del text_encoder_2, output_2, tokens_2
    del tokenizer_1, tokenizer_2
    del prompt_embeds_1, prompt_embeds_2

    clear_memory()

    return prompt_embeds, pooled_prompt_embeds


# ==========================================
# 5. Encode Images with VAE
# ==========================================


def encode_images(source, target):

    print("\n========== VAE Encoding ==========")

    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    vae.eval()

    scaling_factor = vae.config.scaling_factor

    with torch.no_grad():
        source = source.to(
            dtype=torch.float32,
        )

        target = target.to(
            dtype=torch.float32,
        )

        source_latents = vae.encode(source).latent_dist.sample()

        target_latents = vae.encode(target).latent_dist.sample()

        source_latents = source_latents * scaling_factor

        target_latents = target_latents * scaling_factor

    print("Source Latents:", source_latents.shape)

    print("Target Latents:", target_latents.shape)

    del vae

    clear_memory()

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
# 6. Build SDXL Time IDs
# ==========================================


def build_time_ids(batch_size):

    height = IMAGE_SIZE[1]

    width = IMAGE_SIZE[0]

    time_ids = torch.tensor(
        [
            height,
            width,
            0,
            0,
            height,
            width,
        ],
        dtype=WEIGHT_DTYPE,
        device=DEVICE,
    )

    time_ids = time_ids.unsqueeze(0).repeat(
        batch_size,
        1,
    )

    return time_ids


# ==========================================
# 7. Load UNet and Inject LoRA
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

    unet.train()

    # Enable gradient checkpointing.
    unet.enable_gradient_checkpointing()

    print("UNet loaded.")

    print("UNet dtype:", next(unet.parameters()).dtype)

    print("\n========== Injecting LoRA ==========")

    configure_unet_lora(unet)

    # Keep trainable LoRA parameters in FP32.
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    print_trainable_parameters(unet)

    return unet


# ==========================================
# 8. Save LoRA Adapter
# ==========================================


def save_lora_adapter(unet, save_dir):

    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========== Saving LoRA Adapter ==========")

    # Get LoRA configuration.
    lora_config = unet.peft_config["default"]

    # Save adapter configuration.
    lora_config.save_pretrained(save_dir)

    # Extract LoRA parameters only.
    lora_state_dict = get_peft_model_state_dict(
        unet,
        adapter_name="default",
    )

    # Move tensors to CPU.
    lora_state_dict = {
        key: value.detach().cpu().contiguous() for key, value in lora_state_dict.items()
    }

    # Save weights.
    weight_path = save_dir / "adapter_model.safetensors"

    save_file(
        lora_state_dict,
        str(weight_path),
    )

    print("Adapter directory:", save_dir.resolve())

    print("Number of LoRA tensors:", len(lora_state_dict))

    print("Weight file:", weight_path)

    print("LoRA Adapter Saved: PASSED")

    return lora_state_dict


# ==========================================
# 9. Reload LoRA Adapter
# ==========================================


def verify_lora_reload(save_dir):

    print("\n========== Reloading LoRA Adapter ==========")

    save_dir = Path(save_dir)

    # Load configuration.
    lora_config = LoraConfig.from_pretrained(save_dir)

    # Load saved weights.
    saved_state = load_file(
        str(save_dir / "adapter_model.safetensors"),
        device="cpu",
    )

    # Load fresh UNet.
    test_unet = UNet2DConditionModel.from_pretrained(
        MODEL_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    # Inject LoRA.
    test_unet.add_adapter(lora_config)

    for param in test_unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Load adapter weights.
    set_peft_model_state_dict(
        test_unet,
        saved_state,
        adapter_name="default",
    )

    # Extract restored weights.
    reloaded_state = get_peft_model_state_dict(
        test_unet,
        adapter_name="default",
    )

    # Compare.
    if set(saved_state.keys()) != set(reloaded_state.keys()):
        raise RuntimeError("LoRA parameter keys do not match.")

    for key in saved_state:
        if not torch.equal(
            saved_state[key],
            reloaded_state[key].cpu(),
        ):
            raise RuntimeError(f"Parameter mismatch: {key}")

    print("Reloaded tensors:", len(reloaded_state))

    print("LoRA Reload Validation: PASSED")

    del test_unet

    clear_memory()


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(42)

    torch.cuda.manual_seed_all(42)

    print("\n========== Environment ==========")

    print("PyTorch:", torch.__version__)

    print("GPU:", torch.cuda.get_device_name(0))

    print_gpu_memory()

    # --------------------------------------
    # Dataset
    # --------------------------------------

    source, target, prompts = load_training_batch()

    # --------------------------------------
    # Text Encoding
    # --------------------------------------

    prompt_embeds, pooled_prompt_embeds = encode_prompts(prompts)

    # --------------------------------------
    # VAE Encoding
    # --------------------------------------

    source_latents, target_latents = encode_images(
        source,
        target,
    )

    del source, target

    clear_memory()

    # --------------------------------------
    # Scheduler
    # --------------------------------------

    scheduler = DDPMScheduler.from_pretrained(
        MODEL_PATH,
        subfolder="scheduler",
        local_files_only=True,
    )

    # --------------------------------------
    # UNet + LoRA
    # --------------------------------------

    unet = load_unet()

    # --------------------------------------
    # Optimizer
    # --------------------------------------

    trainable_params = [param for param in unet.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("\nOptimizer initialized.")

    print("Trainable tensors:", len(trainable_params))

    # Save initial LoRA parameters.
    initial_lora_params = [param.detach().clone() for param in trainable_params]

    # Save one original UNet weight for comparison.
    base_param = next(
        param for name, param in unet.named_parameters() if "base_layer.weight" in name
    )

    base_param_before = base_param.detach().clone()

    # --------------------------------------
    # Diffusion Noise
    # --------------------------------------

    print("\n========== Diffusion Noise ==========")

    batch_size = target_latents.shape[0]

    noise = torch.randn_like(target_latents)

    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=DEVICE,
    ).long()

    noisy_target = scheduler.add_noise(
        target_latents,
        noise,
        timesteps,
    )

    print("Noise:", noise.shape)

    print("Timesteps:", timesteps)

    print("Noisy Target:", noisy_target.shape)

    # --------------------------------------
    # UNet Input
    # --------------------------------------

    unet_input = torch.cat(
        [
            noisy_target,
            source_latents,
        ],
        dim=1,
    )

    time_ids = build_time_ids(batch_size)

    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": time_ids,
    }

    print("\n========== UNet Input ==========")

    print("UNet Input:", unet_input.shape)

    print("Time IDs:", time_ids.shape)

    # --------------------------------------
    # Training Target
    # --------------------------------------

    prediction_type = scheduler.config.prediction_type

    if prediction_type == "epsilon":
        training_target = noise

    elif prediction_type == "v_prediction":
        training_target = scheduler.get_velocity(
            target_latents,
            noise,
            timesteps,
        )

    elif prediction_type == "sample":
        training_target = target_latents

    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    # --------------------------------------
    # Forward
    # --------------------------------------

    print("\n========== UNet Forward ==========")

    optimizer.zero_grad(set_to_none=True)

    noise_pred = unet(
        unet_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=True,
    ).sample

    print("Noise Prediction:", noise_pred.shape)

    print("Prediction Type:", prediction_type)

    # --------------------------------------
    # Diffusion Loss
    # --------------------------------------

    loss = F.mse_loss(
        noise_pred.float(),
        training_target.float(),
    )

    print("Diffusion Loss:", loss.item())

    if not torch.isfinite(loss):
        raise RuntimeError("Loss contains NaN or Inf.")

    # --------------------------------------
    # Backward
    # --------------------------------------

    print("\n========== Backward ==========")

    loss.backward()

    grad_tensors = [param.grad for param in trainable_params if param.grad is not None]

    if not grad_tensors:
        raise RuntimeError("No LoRA gradients found.")

    if not all(torch.isfinite(g).all().item() for g in grad_tensors):
        raise RuntimeError("LoRA gradients contain NaN or Inf.")

    grad_norm = torch.sqrt(
        sum(torch.sum(g.detach().float() ** 2) for g in grad_tensors)
    )

    if grad_norm.item() == 0:
        raise RuntimeError("LoRA gradients are all zero.")

    # Verify frozen parameters have no gradients.
    invalid_grads = [
        name
        for name, param in unet.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]

    if invalid_grads:
        raise RuntimeError(f"Frozen parameters received gradients: {invalid_grads[:5]}")

    print("Gradient Tensors:", len(grad_tensors))

    print("Gradient Norm:", grad_norm.item())

    print("Frozen Parameter Gradient Check: PASSED")

    # --------------------------------------
    # Optimizer Step
    # --------------------------------------

    print("\n========== Optimizer Step ==========")

    optimizer.step()

    changed_params = 0

    for before, after in zip(
        initial_lora_params,
        trainable_params,
    ):
        if not torch.equal(before, after.detach()):
            changed_params += 1

    if changed_params == 0:
        raise RuntimeError("No LoRA parameters were updated.")

    base_unchanged = torch.equal(
        base_param_before,
        base_param.detach(),
    )

    if not base_unchanged:
        raise RuntimeError("Original UNet weight was modified.")

    print("Changed LoRA Tensors:", changed_params)

    print("Original UNet Weight Unchanged:", base_unchanged)

    # --------------------------------------
    # Final Results
    # --------------------------------------

    print("\n========== Final Results ==========")

    print("Loss:", loss.item())

    print("Gradient Norm:", grad_norm.item())

    print("Changed LoRA Tensors:", changed_params)

    print("Original UNet Frozen: PASSED")

    print("LoRA Backward: PASSED")

    print("LoRA Optimizer Update: PASSED")

    # ======================================
    # Save LoRA Adapter
    # ======================================

    SAVE_DIR = "checkpoints/semantic_lora_single_step"

    saved_state = save_lora_adapter(
        unet,
        SAVE_DIR,
    )

    # ======================================
    # Release GPU Memory
    # ======================================

    del optimizer
    del unet

    clear_memory()

    # ======================================
    # Reload and Verify
    # ======================================

    verify_lora_reload(SAVE_DIR)

    del saved_state

    clear_memory()

    print("\n========== Final Results ==========")

    print("Single Training Step: PASSED")

    print("LoRA Adapter Save: PASSED")

    print("LoRA Adapter Reload: PASSED")

    print("All Validations: PASSED")


if __name__ == "__main__":
    try:
        main()

    except torch.cuda.OutOfMemoryError:
        clear_memory()

        print("\nCUDA OUT OF MEMORY.")

        print("Try reducing IMAGE_SIZE to (256, 128).")

        raise
