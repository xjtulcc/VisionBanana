import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline

from training.lora_config import (
    configure_unet_lora,
    print_trainable_parameters,
)

MODEL_PATH = "models/sdxl-instructpix2pix"


def main():

    print("\n========== Loading SDXL Pipeline ==========")

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print("Pipeline loaded.")

    unet = pipe.unet

    print("\n========== Original UNet ==========")

    print(
        "Total Parameters:",
        sum(p.numel() for p in unet.parameters()),
    )

    print("\n========== Injecting LoRA ==========")

    configure_unet_lora(unet)

    print("LoRA injection completed.")

    print("\n========== Checking Parameters ==========")

    statistics = print_trainable_parameters(unet)

    print("\n========== Validation Results ==========")

    print("Total:", statistics["total_params"])

    print("Trainable:", statistics["trainable_params"])

    print(
        "Trainable Ratio:",
        f"{statistics['trainable_percentage']:.6f}%",
    )

    print("\nLoRA Configuration Test: PASSED")

    print("No optimizer created.")

    print("No backward propagation performed.")

    print("No model parameters updated.")


if __name__ == "__main__":
    main()
