import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    PreTrainedModel
)
from qwen_vl_utils import process_vision_info
from torchvision import transforms
import os


class QwenVLMWrapper(nn.Module):
    """
    Wrapper for Qwen2.5-VL that enables text-only attacks with fixed images.
    
    This wrapper pre-processes images and provides a text-only interface
    that's compatible with controllable-seo attack algorithms.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        
        # Load model and processor
        print(f"🔄 Loading Qwen2.5-VL from {model_path}...")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            **kwargs
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        # Set all model parameters to requires_grad=False for controllable-seo compatibility
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        
        # Cache for pre-processed vision features
        self._fixed_images = None
        self._fixed_pixel_values = None
        self._fixed_image_grid_thw = None
        self._image_prefix_embeddings = None  
        self._image_prefix_input_ids = None   
        
        print("✅ VLM wrapper initialized successfully!")
    
    def set_fixed_images(self, images: List[str]) -> None:

        print(f"🖼️  Setting {len(images)} fixed images...")
        
        # 验证图片路径
        for i, img_path in enumerate(images):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {i+1} not found: {img_path}")
            else:
                print(f"✅ Image {i+1} found: {img_path}")
        
        # 构造特有prompt：Product 1 image: <image1> Product 2 image: <image2> ...
        image_content = []
        for i, img_path in enumerate(images, 1):
            image_content.append({"type": "text", "text": f"Product {i} image: "})
            image_content.append({"type": "image", "image": img_path})
        
        
        # 创建完整的multimodal messages
        messages = [{
                "role": "system",
                "content": "You are a recommendation AI designed to assist users in selecting the best products based on their preferences and needs. Provide a numbered list of product recommendations ranked according to the user's request."
            },
            {
                "role": "user",
                "content": image_content
            }]
        
        # 处理图像
        image_inputs, video_inputs = process_vision_info(messages)
        
        to_tensor = transforms.ToTensor()
        tensor_image_inputs = [to_tensor(pil_img).to(self.device) for pil_img in image_inputs]
        
        # 使用processor处理完整的multimodal输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # 删除结尾的<|im_end|>，以便后续拼接更多文本内容
        if text.endswith('<|im_end|>\n'):
            text = text[:-len('<|im_end|>\n')]
        elif text.endswith('<|im_end|>'):
            text = text[:-len('<|im_end|>')]
        
        text = text + '\n'
            
        # print(f"🔍 Text: {text}")
        processed_inputs = self.processor(
            text=[text],
            images=tensor_image_inputs,  
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        processed_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in processed_inputs.items()}
        
        # 缓存所有组件
        self._fixed_images = images
        self._fixed_pixel_values = processed_inputs.get('pixel_values')
        self._fixed_image_grid_thw = processed_inputs.get('image_grid_thw')
        
        # 缓存图像前缀的input_ids和embeddings
        input_ids = processed_inputs['input_ids']
        self._image_prefix_input_ids = input_ids
        
        # 生成图像前缀的embeddings（这是关键！）
        embedding_layer = self.get_input_embeddings()
        self._image_prefix_embeddings = embedding_layer(input_ids).detach()
        
        print(f"✅ Images cached successfully!")
    
    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer of the text model.
        Required for controllable-seo compatibility.
        """
        # For Qwen2.5-VL, access through the VLM's get_input_embeddings method
        return self.vlm_model.get_input_embeddings()
    
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_image_prefix: bool = False,
        **kwargs
    ) -> Any:
        """
        Forward pass supporting both input_ids and inputs_embeds.
        Compatible with controllable-seo's soft_forward mechanism.
        
        Args:
            input_ids: Token IDs (mutually exclusive with inputs_embeds)
            inputs_embeds: Soft embeddings for adversarial attacks
            labels: Labels for loss calculation
            use_cache: Whether to use KV cache
            use_image_prefix: Whether to concatenate image prefix tokens (default: False)
            
        Returns:
            Model outputs
        """
        # If we have images cached, we need to handle multimodal inputs
        if self._fixed_pixel_values is not None:
            if inputs_embeds is not None:
                
                if self._image_prefix_embeddings is None:
                    raise ValueError("No image prefix embeddings cached. Call set_fixed_images first.")
                

                full_embeddings = torch.cat([self._image_prefix_embeddings, inputs_embeds], dim=1)
                
                full_seq_length = full_embeddings.shape[1]
                full_attention_mask = torch.ones(inputs_embeds.shape[0], full_seq_length,
                                               dtype=torch.long, device=inputs_embeds.device)
                
                outputs = self.vlm_model(
                    inputs_embeds=full_embeddings,
                    pixel_values=self._fixed_pixel_values,        
                    image_grid_thw=self._fixed_image_grid_thw,    
                    attention_mask=full_attention_mask,
                    labels=labels,
                    use_cache=use_cache,
                    **kwargs
                )
                
                
                if hasattr(outputs, 'logits'):
                    image_prefix_length = self._image_prefix_embeddings.shape[1]
                    outputs.logits = outputs.logits[:, image_prefix_length:, :]
                
                return outputs
            
            elif input_ids is not None:
                if use_image_prefix:
                    if self._image_prefix_input_ids is None:
                        raise ValueError("No image prefix input_ids cached. Call set_fixed_images first.")
                    
                    full_input_ids = torch.cat([self._image_prefix_input_ids, input_ids], dim=1)
                    
                    full_seq_length = full_input_ids.shape[1]
                    full_attention_mask = torch.ones(input_ids.shape[0], full_seq_length,
                                                   dtype=torch.long, device=input_ids.device)
                                                   
                    outputs = self.vlm_model(
                        input_ids=full_input_ids,
                        pixel_values=self._fixed_pixel_values,        
                        image_grid_thw=self._fixed_image_grid_thw,    
                        attention_mask=full_attention_mask,
                        labels=labels,
                        use_cache=use_cache,
                        **kwargs
                    )
                    
                    if hasattr(outputs, 'logits'):
                        image_prefix_length = self._image_prefix_input_ids.shape[1]
                        outputs.logits = outputs.logits[:, image_prefix_length:, :]
                    
                    return outputs
                else:
                    attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1],
                                             dtype=torch.long, device=input_ids.device)
                    
                    outputs = self.vlm_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=use_cache,
                        **kwargs
                    )
                    
                    return outputs
            else:
                # Neither input_ids nor inputs_embeds provided
                raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        else:
            # No images cached - this wrapper requires images to be set
            raise ValueError(
                "No images have been set. This VLM wrapper requires images to be configured. "
                "Please call set_fixed_images() first."
            )
    
    def generate(self, input_ids=None, **kwargs) -> torch.Tensor:
        """
        Generation method for inference.
        Compatible with controllable-seo's calling pattern.
        """
        if input_ids is not None:
            kwargs['input_ids'] = input_ids
        return self.vlm_model.generate(**kwargs)
    
    def eval(self):
        """
        Set model to evaluation mode.
        """
        self.vlm_model.eval()
        return self
    
    
    @property
    def config(self):
        """Access to model config."""
        return self.vlm_model.config
    
    @property
    def generation_config(self):
        """Access to model generation config."""
        return self.vlm_model.generation_config
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size


