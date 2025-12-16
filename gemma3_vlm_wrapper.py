import torch
import torch.nn as nn
from typing import List, Optional, Any
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
from torchvision import transforms
import os
from qwen_vl_utils import process_vision_info
class Gemma3VLMWrapper(nn.Module):
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
        
        print(f"🔄 Loading Gemma-3-12B-IT from {model_path}...")
        self.vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            **kwargs
        ).eval()
        
        self.vlm_model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for memory optimization")
        
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        
        self._fixed_images = None
        self._pixel_values = None
        self._image_grid_thw = None
        self._image_prefix_input_ids = None
        self._image_prefix_embeddings = None
        
        print("✅ Gemma-3 wrapper initialized successfully!")
    
    def set_fixed_images(self, images: List[str]) -> None:
        print(f"🖼️  Setting {len(images)} fixed images...")
        
        for i, img_path in enumerate(images):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {i+1} not found: {img_path}")
        
        image_content = []
        for i, img_path in enumerate(images, 1):
            image_content.append({"type": "text", "text": f"Product {i} image: "})
            image_content.append({"type": "image", "image": img_path})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a recommendation AI designed to assist users in selecting the best products based on their preferences and needs. Provide a numbered list of product recommendations ranked according to the user's request."}]},
            {"role": "user", "content": image_content}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        if text.endswith('<end_of_turn>\n'):
            text = text[:-len('<end_of_turn>\n')]
        elif text.endswith('<end_of_turn>'):
            text = text[:-len('<end_of_turn>')]
        
        #print(f"✅ Text: {text}")

        image_inputs, _ = process_vision_info(messages)
        to_tensor = transforms.ToTensor()
        tensor_image_inputs = [to_tensor(pil_img) for pil_img in image_inputs]
        
        
        processed_inputs = self.processor(
            text=[text],
            images=tensor_image_inputs,
            return_tensors="pt"
        )
        
        processed_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in processed_inputs.items()}
        
        self._fixed_images = images
        self._pixel_values = processed_inputs['pixel_values']
        self._image_grid_thw = processed_inputs.get('image_grid_thw')
        self._image_prefix_input_ids = processed_inputs['input_ids']
        
        embedding_layer = self.get_input_embeddings()
        self._image_prefix_embeddings = embedding_layer(self._image_prefix_input_ids).detach()
        
        print(f"✅ Images cached successfully!")
        print(f"   Image prefix length: {self._image_prefix_input_ids.shape[1]} tokens")
    
    def get_input_embeddings(self) -> nn.Module:
        return self.vlm_model.get_input_embeddings()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_image_prefix: bool = False,
        **kwargs
    ) -> Any:
        if self._pixel_values is not None:
            if inputs_embeds is not None:
                if self._image_prefix_embeddings is None:
                    raise ValueError("No image prefix embeddings cached.")
                
                full_embeddings = torch.cat([self._image_prefix_embeddings, inputs_embeds], dim=1)
                
                full_attention_mask = torch.ones(
                    inputs_embeds.shape[0], 
                    full_embeddings.shape[1],
                    dtype=torch.long, 
                    device=inputs_embeds.device
                )
                
                outputs = self.vlm_model(
                    inputs_embeds=full_embeddings,
                    pixel_values=self._pixel_values,
                    image_grid_thw=self._image_grid_thw,
                    attention_mask=full_attention_mask,
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
                        raise ValueError("No image prefix input_ids cached.")
                    
                    full_input_ids = torch.cat([self._image_prefix_input_ids, input_ids], dim=1)
                    
                    full_attention_mask = torch.ones(
                        input_ids.shape[0],
                        full_input_ids.shape[1],
                        dtype=torch.long,
                        device=input_ids.device
                    )
                    
                    outputs = self.vlm_model(
                        input_ids=full_input_ids,
                        pixel_values=self._pixel_values,
                        image_grid_thw=self._image_grid_thw,
                        attention_mask=full_attention_mask,
                        use_cache=use_cache,
                        **kwargs
                    )
                    
                    if hasattr(outputs, 'logits'):
                        image_prefix_length = self._image_prefix_input_ids.shape[1]
                        outputs.logits = outputs.logits[:, image_prefix_length:, :]
                    
                    return outputs
                else:
                    attention_mask = torch.ones(
                        input_ids.shape[0],
                        input_ids.shape[1],
                        dtype=torch.long,
                        device=input_ids.device
                    )
                    
                    outputs = self.vlm_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        **kwargs
                    )
                    
                    return outputs
            else:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
        else:
            raise ValueError("No images set. Call set_fixed_images() first.")
    
    def generate(self, input_ids=None, **kwargs) -> torch.Tensor:
        if input_ids is not None:
            kwargs['input_ids'] = input_ids
        return self.vlm_model.generate(**kwargs)
    
    def eval(self):
        self.vlm_model.eval()
        return self
    
    @property
    def config(self):
        return self.vlm_model.config
    
    @property
    def generation_config(self):
        return self.vlm_model.generation_config
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size



