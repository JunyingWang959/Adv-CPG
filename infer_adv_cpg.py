import argparse
import torch
from PIL import Image
from pipline_AdvCPG_StableDiffusion import AdvCPGStableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
import numpy as np

def infer(
    base_model: str,
    advcpg_ckpt: str,
    bise_net_cp: str,
    image_encoder: str,
    fr_models: list,
    llava_model: str,
    prompt: str,
    ori_path: str,
    target_path: str,
    out_path: str,
    num_tokens: int,
    lora_rank: int,
    device: str = "cuda",
):
    # 1) 加载 Adv-CPG Pipeline
    pipe = AdvCPGStableDiffusionXLPipeline.from_pretrained(
        base_model,
        safety_checker=None,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    # 2) 加载权重
    pipe.load_adv_cpg_model(
        base_model=base_model,
        advcpg_ckpt=advcpg_ckpt,
        bise_net_cp=bise_net_cp,
        image_encoder_path=image_encoder,
        fr_model_paths=fr_models,
        llava_model_path=llava_model,
        num_tokens=num_tokens,
        lora_rank=lora_rank,
    )

    # 3) 调度器
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # 4) 读取图像
    source_image = load_image(ori_path)
    target_image = load_image(target_path)

    # 5) 推理
    result = pipe(
        prompt=prompt,
        source_image=source_image,
        target_image=target_image,
        source_image_path=ori_path,
        target_image_path=target_path,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=torch.Generator(device=device).manual_seed(42),
        return_dict=False,
    )[0]
    # 因为 return_dict=False，result 就是 PIL.Image
    result.save(out_path)
    print(f"✅ Saved adv-cpg result at {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",       type=str, default="models--stabilityai--stable-diffusion-xl-base-1.0")
    p.add_argument("--advcpg_ckpt", type=str, default="ConsistentID_SDXL-v1.bin", help="you can also use the weight of ConsistentID to have a quick start.")
    p.add_argument("--bise_net_cp",   type=str, default="models/face_parsing.pth")
    p.add_argument("--image_encoder",    type=str, default="CLIP-ViT-H-14-laion2B-s32B-b79K")
    p.add_argument("--fr_models",        nargs="+", default=[
        "models/ir152.pth",
        "models/irse50.pth",
        "models/facenet.pth",
        "models/mobileface.pth",
    ])
    p.add_argument("--llava_model",      type=str, default=None)
    p.add_argument("--prompt", type=str, default="A photo of a person in a classroom, cinematic lighting",
                        help="The text prompt for customization.")
    p.add_argument("--ori_path", type=str, default="./examples/scarlett_johansson.jpg",
                        help="path of the source person (for visual appearance).")
    p.add_argument("--target_path", type=str, default="./examples/albert_einstein.jpg",
                        help="path of the target person (for adversarial identity).")
    p.add_argument("--out", type=str, default="adv_cpg_out.png")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_tokens",  type=int, default=4,  help="trigger token 数")
    p.add_argument("--lora_rank",   type=int, default=128,  help="LoRA rank")
    args = p.parse_args()


    infer(
        base_model=args.base_model,
        advcpg_ckpt=args.advcpg_ckpt,
        bise_net_cp=args.bise_net_cp,
        image_encoder=args.image_encoder,
        fr_models=args.fr_models,
        llava_model=args.llava_model,
        prompt=args.prompt,
        ori_path=args.ori_path,
        target_path=args.target_path,
        out_path=args.out,
        num_tokens=args.num_tokens,
        lora_rank=args.lora_rank,
        device=args.device,
    )