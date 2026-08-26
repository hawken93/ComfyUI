import os

import comfy.sd

def first_file(path, filenames):
    for f in filenames:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return p
    return None

def load_diffusers(device: str, model_path, output_vae=True, output_clip=True, embedding_directory=None):
    # `device` is the raw user device option string from the loader node
    # ("default", "cpu", or a concrete device id like "cuda:0"), not a
    # resolved torch.device. It is resolved with pick_device_for_option
    # next to the VAE state dict load, where its size and dtype are
    # known, so the default lands on a device that supports the dtype and
    # has room for the weights; the unet then loads onto the same device.
    diffusion_model_names = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.bin", "diffusion_pytorch_model.bin"]
    unet_path = first_file(os.path.join(model_path, "unet"), diffusion_model_names)
    vae_path = first_file(os.path.join(model_path, "vae"), diffusion_model_names)

    text_encoder_model_names = ["model.fp16.safetensors", "model.safetensors", "pytorch_model.fp16.bin", "pytorch_model.bin"]
    text_encoder1_path = first_file(os.path.join(model_path, "text_encoder"), text_encoder_model_names)
    text_encoder2_path = first_file(os.path.join(model_path, "text_encoder_2"), text_encoder_model_names)

    text_encoder_paths = [text_encoder1_path]
    if text_encoder2_path is not None:
        text_encoder_paths.append(text_encoder2_path)

    # Budget: the unet's param count isn't known before load, so use its file
    # size as the footprint proxy (a safetensors file is ~exactly the
    # weights); take the max with the vae's known size.
    memory_required = os.path.getsize(unet_path)
    dtype = None
    if output_vae:
        vae_sd = comfy.utils.load_torch_file(vae_path)
        dtype = comfy.utils.weight_dtype(vae_sd)
        memory_required = max(memory_required, comfy.utils.calculate_parameters(vae_sd) * comfy.model_management.dtype_size(dtype))

    resolved_device = comfy.model_management.pick_device_for_option(device, memory_required=memory_required, dtype=dtype)
    unet_offload = comfy.model_management.unet_offload_device(resolved_device)
    unet = comfy.sd.load_diffusion_model(unet_path, resolved_device, unet_offload)

    clip = None
    if output_clip:
        clip = comfy.sd.load_clip(text_encoder_paths, embedding_directory=embedding_directory)

    vae = None
    if output_vae:
        vae = comfy.sd.VAE(resolved_device, sd=vae_sd)

    return (unet, clip, vae)
