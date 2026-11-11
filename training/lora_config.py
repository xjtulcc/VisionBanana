from peft import LoraConfig

# ==========================================
# 1. LoRA Configuration
# ==========================================

LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]


def build_lora_config():
    """
    Build LoRA configuration for SDXL UNet.
    """

    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        init_lora_weights=True,
    )

    return config


# ==========================================
# 2. Inject LoRA into UNet
# ==========================================


def configure_unet_lora(unet):
    """
    Freeze original UNet parameters
    and inject trainable LoRA adapters.
    """

    # Freeze original UNet parameters.
    unet.requires_grad_(False)

    # Build LoRA configuration.
    lora_config = build_lora_config()

    # Inject LoRA adapters.
    unet.add_adapter(lora_config)

    return unet


# ==========================================
# 3. Parameter Statistics
# ==========================================


def print_trainable_parameters(model):
    """
    Print total and trainable parameter statistics.
    """

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100.0 * trainable_params / total_params

    print("\n========== LoRA Parameter Statistics ==========")

    print(f"Total Parameters: {total_params:,}")

    print(f"Trainable Parameters: {trainable_params:,}")

    print(f"Frozen Parameters: {total_params - trainable_params:,}")

    print(f"Trainable Percentage: {percentage:.6f}%")

    print("\n========== Trainable Parameter Names ==========")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(
                name,
                tuple(param.shape),
            )

    # Ensure only LoRA parameters are trainable.
    invalid_params = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]

    if invalid_params:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid_params}")

    if trainable_params == 0:
        raise RuntimeError("No trainable LoRA parameters found.")

    print("\nLoRA Parameter Validation: PASSED")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": percentage,
    }
