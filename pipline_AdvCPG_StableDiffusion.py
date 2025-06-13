import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Any, List, Optional, Union

from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms
from insightface.app import FaceAnalysis

from pipline_StableDiffusionXL_ConsistentID import ConsistentIDStableDiffusionXLPipeline


class AdvCPGStableDiffusionXLPipeline(ConsistentIDStableDiffusionXLPipeline):
    def load_adv_cpg_model(
        self,
        base_model: str,
        advcpg_ckpt: str,
        bise_net_cp: str,
        image_encoder_path: str,
        fr_model_paths: List[str],
        llava_model_path: Optional[str] = None,
        torch_dtype=torch.float16,
        num_tokens: int = 4,
        lora_rank: int = 128,
    ):
        # 1) 加载权重
        self.num_tokens = num_tokens
        self.lora_rank = lora_rank
        ckpt_dir, weight_name = os.path.split(advcpg_ckpt)
        super().load_ConsistentID_model(
            pretrained_model_name_or_path_or_dict=ckpt_dir,
            weight_name=weight_name,
            subfolder="",
            trigger_word_ID="<|image|>",
            trigger_word_facial="<|facial|>",
            image_encoder_path=image_encoder_path,
            bise_net_cp=bise_net_cp,
            torch_dtype=torch_dtype,
            num_tokens=self.num_tokens,
            lora_rank=self.lora_rank,
        )

        # 2) CLIP 编码器 (for get_Cid)
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(self.device, dtype=torch_dtype)
        self.clip_image_processor = CLIPImageProcessor()

        # 3) InsightFace FR 模型
        self.fr_apps = []
        for fr_path in fr_model_paths:
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(512,512))
            self.fr_apps.append(app)

        # 4) FR->CLIP dim projection
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=torch_dtype)
            out = self.clip_image_encoder(dummy, output_hidden_states=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                clip_dim = out.pooler_output.shape[-1]
            elif hasattr(out, "pooled_output") and out.pooled_output is not None:
                clip_dim = out.pooled_output.shape[-1]
            else:
                clip_dim = out.last_hidden_state.shape[-1]
        self.fr_proj = nn.Linear(512, clip_dim).to(self.device, dtype=torch_dtype)

        # 5) LLaVA fallback
        self.llava_model_path = llava_model_path
        self.default_face_prompt = (
            "Describe this person's facial features, including face shape, eyes, nose, mouth, hair, etc."
        )

        return self

    @torch.no_grad()
    def get_Cid(self, target_image: Image.Image):
        # CLIP features
        clip_inputs = self.clip_image_processor(images=target_image, return_tensors="pt").pixel_values
        clip_inputs = clip_inputs.to(self.device, dtype=self.vae.dtype)
        clip_out = self.clip_image_encoder(clip_inputs, output_hidden_states=True)
        if hasattr(clip_out, "pooler_output") and clip_out.pooler_output is not None:
            clip_feats = clip_out.pooler_output
        elif hasattr(clip_out, "pooled_output") and clip_out.pooled_output is not None:
            clip_feats = clip_out.pooled_output
        else:
            clip_feats = clip_out.last_hidden_state[:, 0, :]

        # FR features
        fr_feats = []
        raw = np.array(target_image)[:, :, ::-1].copy()
        for app in self.fr_apps:
            infos = app.get(raw)
            if infos:
                fr_feats.append(torch.from_numpy(infos[0].normed_embedding).to(self.device))
        if len(fr_feats) > 0:
            fr_feats = torch.stack(fr_feats).mean(0, keepdim=True)
        else:
            fr_feats = torch.zeros(1, 512, device=self.device)
        fr_feats = fr_feats.to(self.device, dtype=clip_feats.dtype)
        fr_feats = self.fr_proj(fr_feats)

        α, β = 0.5, 0.1
        return α * clip_feats + β * fr_feats  # [1, clip_dim]

    @torch.no_grad()
    def get_adversarial_image_embeds(
        self,
        target_image: Image.Image,
        target_image_path: str,
        s_scale: float = 1.0,
        shortcut: bool = True,
    ):
        """
        Adv-CPG: 直接用 InsightFace embedding (512-d) 注入对抗身份
        """
        faceid_embeds = self.get_prepare_faceid(input_image_path=target_image_path)
        if faceid_embeds is None:
            raise ValueError("无法提取目标人脸 embedding")
        return self.get_image_embeds(
            faceid_embeds, face_image=target_image, s_scale=s_scale, shortcut=shortcut
        )

    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        source_image: Any,
        target_image: Any,
        source_image_path: str,
        target_image_path: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        # prep
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor
        device = self._execution_device
        do_cfg = guidance_scale >= 1.0

        # 1) 文本编码
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            self.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
            )

        # 2) SOURCE IP-Adapter 注入
        src_feats = self.get_prepare_faceid(input_image_path=source_image_path)
        id_tokens, uncond_id_tokens = self.get_image_embeds(
            src_feats, face_image=source_image, s_scale=1.5, shortcut=True
        )

        # 3) TARGET IP-Adapter 注入
        adv_tokens, uncond_adv_tokens = self.get_adversarial_image_embeds(
            target_image=target_image,
            target_image_path=target_image_path,
            s_scale=0.6,
            shortcut=True,
        )

        # 4) 拼接
        prompt_embeds = torch.cat([prompt_embeds, id_tokens, adv_tokens], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_id_tokens, uncond_adv_tokens], dim=1)

        # 5) scheduler & latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_size = prompt_embeds.shape[0]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height, width,
            prompt_embeds.dtype,
            device, generator, None
        )
        extra_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6) SDXL text_time
        add_time_ids = self._get_add_time_ids(
            (height, width), (0,0), (height, width),
            dtype=pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        ).to(device)

        # 7) denoise
        for t in timesteps:
            # a) 准备 latent 输入（CFG: cat）
            latent_in = torch.cat([latents] * 2) if do_cfg else latents
            latent_in = self.scheduler.scale_model_input(latent_in, t)

            # b) encoder_hidden_states
            if do_cfg:
                encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            else:
                encoder_hidden_states = prompt_embeds

            # c) 扩展 IP-Adapter 的 id_embeds
            #    id_tokens shape [B, num_tokens, clip_dim]
            #    对于 CFG, 扩到 [2B, num_tokens, clip_dim]
            if do_cfg:
                id_ca = torch.cat([id_tokens, id_tokens], dim=0)
            else:
                id_ca = id_tokens

            # d) SDXL text_time 的两个额外输入
            if do_cfg:
                txt = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                tm  = torch.cat([add_time_ids, add_time_ids], dim=0)
            else:
                txt = pooled_prompt_embeds
                tm  = add_time_ids

            # e) 一起放到 added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": txt,    # [B*2, text_proj_dim] or [B, ...]
                "time_ids":    tm,     # [B*2, time_dim]     or [B, ...]
                "id_embeds":   id_ca.unsqueeze(1),  # [B*2, 1, clip_dim] or [B,1,clip_dim]
            }

            # f) 调用 UNet（删除 cross_attention_kwargs）
            noise_pred = self.unet(
                latent_in,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # g) classifier-free guidance
            if do_cfg:
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond - uncond)

            # h) step
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample


        # 8) decode & postprocess (全精度解码以避免 fp16 溢出)
        # 8.1 clamp to safe range
        latents = torch.clamp(latents, -1.0, 1.0)
        # 8.2 prepare input
        dec_in = latents / self.vae.config.scaling_factor

        # 8.3 切换 VAE 到 float32
        prev_dtype = next(self.vae.parameters()).dtype
        self.vae.to(torch.float32)
        # 解码时全部转为 float32
        dec_images = self.vae.decode(dec_in.float(), return_dict=False)[0]  # [B, C, H, W]
        # 恢复 VAE 原始 dtype
        self.vae.to(prev_dtype)

        # 8.4 后处理到 PIL
        pil_images = self.image_processor.postprocess(dec_images, output_type="pil")

        if kwargs.get("return_dict", True):
            return StableDiffusionXLPipelineOutput(images=pil_images)
        return pil_images
