"""
Multi-Model VLM Product Reranker
Supports: Qwen2.5-VL, Gemma-3-12b-it
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
import random
from image_util import generate_image_mask
import traceback
from PIL import Image
import numpy as np
import re
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from qwen_vl_utils import process_vision_info
from attack import attack_control
from attack_autodan import autodan_attack_control
from process import process_bad_words, greedy_decode, init_prompt, process_text, process_stop_words, get_original_embedding
from attack_config import (
    SYSTEM_PROMPT, GUIDING_SENTENCES, BAD_WORDS, ASSSISTANT_PROMPT,
    get_attack_params, get_autodan_params, get_user_query, get_image_attack_params,
    get_multimodal_attack_params
)

def seed_everything(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultimodalProductReranker:
    def __init__(self, model_path, model_name, catalog='baby_strollers', dataset='amazon'):

        self.model_path = model_path
        self.model_name = model_name
        self.catalog = catalog
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Loading Multi-Model VLM Product Reranker...")
        print(f"📁 Model path: {model_path}")
        print(f"🤖 Model type: {model_name}")
        print(f"📊 Catalog: {catalog}")
        print(f"💾 Dataset: {dataset}")
        

        self._load_model()
        self.product_list = self._load_multimodal_products()
        print(f"✅ System ready with {len(self.product_list)} products")
    
    def _load_model(self):
        """Load model based on model_name"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"❌ Model path not found: {self.model_path}")
            
            if self.model_name == "qwen2.5-vl":
                print("🔄 Loading Qwen2.5-VL model...")
                from qwen_vlm_wrapper import QwenVLMWrapper
                self.model = QwenVLMWrapper(
                    model_path=self.model_path,
                    device="cuda",
                    torch_dtype=torch.bfloat16
                )
                self.processor = self.model.processor
                print("✅ Qwen2.5-VL loaded successfully!")
                
            elif self.model_name == "llama-3.2-vision": # not supported for now
                print("🔄 Loading Llama-3.2-Vision model...")
                from llama32_vlm_text_attack_wrapper import Llama32VLMTextAttackWrapper
                self.model = Llama32VLMTextAttackWrapper(
                    model_path=self.model_path,
                    device="cuda",
                    torch_dtype=torch.bfloat16
                )
                self.processor = self.model.processor
                print("✅ Llama-3.2-Vision loaded successfully!")
                
            elif self.model_name == "gemma-3-12b-it":
                print(f"🔄 Loading Gemma-3-12B-IT model...")
                from gemma3_vlm_wrapper import Gemma3VLMWrapper
                self.model = Gemma3VLMWrapper(
                    model_path=self.model_path,
                    device="cuda",
                    torch_dtype=torch.bfloat16
                )
                self.processor = self.model.processor
                print(f"✅ Gemma-3-12B-IT loaded successfully!")
                
            else:
                raise ValueError(f"❌ Unsupported model type: {self.model_name}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")
    
    def _generate_adversarial_text(self, products, target_product_idx, initial_prompt_logits=None, lr=None):
        # Generate adversarial text suffix using the same method as in StealthRank
        print(f"\n🎯 Generating adversarial text for Product {target_product_idx + 1}")
        
        attack_params = get_attack_params()
        
        if lr is not None:
            attack_params['lr'] = lr
            print(f"✅ Using custom learning rate: {lr}")
        else:
            print(f"✅ Using default learning rate: {attack_params['lr']}")
        
        user_msg = get_user_query(self.catalog, len(products))
        
        if self.model_name == "qwen2.5-vl":
            system_prompt = SYSTEM_PROMPT['qwen']
        elif self.model_name == "llama-3.2-vision":
            system_prompt = SYSTEM_PROMPT['llama-3.2-vision']
        elif self.model_name == "gemma-3-12b-it":
            system_prompt = SYSTEM_PROMPT['gemma-3']
        else:
            raise ValueError(f"❌ Unsupported model for text attack: {self.model_name}")
        
        # Get guiding sentence for suffix mode
        guiding_sentence = GUIDING_SENTENCES['suffix']
        
        # Prepare target string (first 3 words of product name as target)
        target_product = products[target_product_idx]['Name']
        product_words = target_product.split()
        first_n_words = ' '.join(product_words[:4])
        target_str = f"1. **{first_n_words}"
        print(f"📊 Target text for text attack: {target_str}")

        # Initialize prompt logits 
        if initial_prompt_logits is None:
            prompt_logits = init_prompt(
                model=self.model, 
                tokenizer=self.processor.tokenizer, 
                product_list=products, 
                target_product_idx=target_product_idx, 
                guiding_sentence=guiding_sentence,
                prompt_length=attack_params['length'], 
                batch_size=attack_params['batch_size'], 
                device=self.device
            )
            print(f"✅ Initialized new prompt logits")
        else:
            prompt_logits = initial_prompt_logits
            print(f"✅ Continuing from previous prompt logits")
        
        # Process target tokens
        target_tokens = process_text(
            tokenizer=self.processor.tokenizer, 
            text=target_str, 
            batch_size=attack_params['batch_size'], 
            device=self.device
        )
        
        # Process bad words tokens 
        extra_word_tokens = process_bad_words(
            bad_words=BAD_WORDS, 
            tokenizer=self.processor.tokenizer, 
            device=self.device
        )
        
        # Decode and print initial prompt 
        _, decoded_init_prompt = greedy_decode(logits=prompt_logits, tokenizer=self.processor.tokenizer)
        print(f"\n📝 DECODED INIT PROMPT: {decoded_init_prompt}")
        
        decoded_prompt, selected_ids, last_y_logits = attack_control(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            system_prompt=system_prompt,
            user_msg=user_msg,
            prompt_logits=prompt_logits,
            target_tokens=target_tokens,
            extra_word_tokens=extra_word_tokens,
            product_list=products,
            target_product=target_product,
            original_embedding=None,  
            **attack_params
        )
        
        return decoded_prompt, selected_ids, last_y_logits
    
    def _generate_adversarial_text_raf(self, products, target_product_idx):
        """Generate adversarial text prompt using autodan_attack_control"""
        #Not used for now
        print(f"\n🎯 Generating adversarial text (RAF) for Product {target_product_idx + 1}")
        
        # Get AutoDAN attack parameters
        autodan_params = get_autodan_params()
        
        # Get user query (with product count for exact instruction matching)
        user_msg = get_user_query(self.catalog, len(products))
        
        # Get system prompt based on model type
        if self.model_name == "qwen2.5-vl":
            system_prompt = SYSTEM_PROMPT['qwen']
        elif self.model_name == "llama-3.2-vision":
            system_prompt = SYSTEM_PROMPT['llama-3.2-vision']
        elif self.model_name == "gemma-3-12b-it":
            system_prompt = SYSTEM_PROMPT['gemma-3']
        else:
            raise ValueError(f"❌ Unsupported model for RAF attack: {self.model_name}")
        
        # Prepare target string (first 3 words of product name as target)
        target_product = products[target_product_idx]['Name']
        product_words = target_product.split()
        first_n_words = ' '.join(product_words[:3])
        target_str = f"1. **{first_n_words}"
        print(f"📊 Target string: {target_str}")
        
        # Create target_token_lists with only one simple template [TARGET]
        target_tokens = self.processor.tokenizer(
            target_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )["input_ids"].to(self.device)
        target_token_lists = [target_tokens]  
        
        # Tokenize target product name
        target_product_tokens = self.processor.tokenizer(
            target_product,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)
        
        print(f"\n🔧 Running AutoDAN attack with {autodan_params['n_steps']} steps...")
        print(f"   Max length: {autodan_params['max_length']}")
        print(f"   Top-k: {autodan_params['topk']}")
        
        # Call autodan_attack_control
        decoded_prompt = autodan_attack_control(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            system_prompt=system_prompt,
            user_msg=user_msg,
            target_token_lists=target_token_lists,
            product_list=products,
            target_product=target_product,
            target_product_tokens=target_product_tokens,
            logger=None,
            table=None,
            **autodan_params
        )
        
        return decoded_prompt
    
    
    def _load_multimodal_products(self):
        """Load product data - only products with valid images"""
        data_file = f'data_new_simplified/{self.dataset}/{self.catalog}.jsonl'
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ Data file not found: {data_file}")
        
        products = []
        excluded_count = 0
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    product = json.loads(line.strip())
                    if 'Name' not in product:
                        continue
                    
                    # Verify image exists
                    if 'image_path' in product:
                        image_path = f'data_new_simplified/{self.dataset}/images/{product["image_path"]}'
                        if os.path.exists(image_path):
                            products.append(product)
                        else:
                            excluded_count += 1
                            print(f"⚠️  Image not found: {image_path}")
                    else:
                        excluded_count += 1
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON at line {line_num}")
                    continue
        
        if excluded_count > 0:
            print(f"📋 Excluded {excluded_count} products without valid images")
            
        print(f"✅ Loaded {len(products)} products with verified images")
        return products
    
    def _process_images(self, products):
        """Process product images - all products must have images"""
        images = []
        
        for product in products:
            try:
                image_path = f'data_new_simplified/{self.dataset}/images/{product["image_path"]}'
                image = Image.open(image_path).convert('RGB')
                images.append(image_path)  
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load image for {product['Name']}: {e}")
        
        return images
    
    def _create_baseline_prompt(self, products, image_paths):
        """Create baseline prompt with all images at the top, then text descriptions"""
        
        content = []
        
        # First, add all images at the top: Product 1 image: <image1> Product 2 image: <image2> ...
        for i, image_path in enumerate(image_paths, 1):
            content.append({"type": "text", "text": f"Product {i} image: "})
            content.append({"type": "image", "image": image_path})
        
        all_product_text = "\n"
        
        # Then add product text information
        for i, product in enumerate(products, 1):
            name = product['Name']
            desc = product.get('Description', 'No description available.')
            
            # Filter out rating information from description
            desc = self._clean_description(desc)
            # Add product text details
            all_product_text += f"Product {i}:\nName: {name}\nDescription: {desc}\n\n"
        
        # Add final ranking instruction
        instruction = get_user_query(self.catalog, len(products))
        all_product_text += instruction
        
        content.append({"type": "text", "text": all_product_text})
        
        return content
    
    def _clean_description(self, description):
        """Clean description by removing only rating information"""
        
        if not description or description == 'No description available.':
            return description
        
        # Remove only rating patterns like "| Rating: 4.0 (123 reviews) |"
        # Pattern matches: | Rating: ... |
        pattern = r'\|\s*Rating:\s*[^|]*\|\s*'
        cleaned = re.sub(pattern, '| ', description, flags=re.IGNORECASE)
        
        # Clean up any leading/trailing whitespace and multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        
        return cleaned
    
    
    def _tv_loss_perturbation(self, delta):
        dh = (delta[:, 1:, :] - delta[:, :-1, :]).pow(2).mean()
        dw = (delta[:, :, 1:] - delta[:, :, :-1]).pow(2).mean()
        
        return dh + dw
    
    def _magnitude_loss_perturbation(self, delta, background_mask=None, object_weight=1.0):
                
        background_region = (background_mask == 1.0)
        object_region = ~background_region
        
        weight_mask = background_region.float() * 1.0 + object_region.float() * object_weight
        
        weighted_delta = torch.abs(delta) * weight_mask
        return weighted_delta.sum() / weight_mask.sum()
    
    def _generate_adversarial_image(self, products, image_paths, target_product_idx, 
                                   global_original_tensor=None, background_mask=None, alpha=None):
        image_attack_params = get_image_attack_params()
        epsilon = image_attack_params['epsilon']
        if alpha is None:
            alpha = image_attack_params['alpha']
        num_steps = image_attack_params['num_steps']
        precision = image_attack_params['precision']
        object_perturbation_ratio = image_attack_params['object_perturbation_ratio']
        smoothness = image_attack_params.get('smoothness', 0.0)
        magnitude = image_attack_params.get('magnitude', 0.0)
        object_magnitude_weight = image_attack_params.get('object_magnitude_weight', 1.0)
        
        print(f"\n🎯 Generating adversarial image for Product {target_product_idx + 1}")
        print(f"📊 PGD parameters: epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
        
        is_first_round = (global_original_tensor is None)
        if is_first_round:
            print(f"✅ First round: will save original tensor and mask for future rounds")
        else:
            print(f"✅ Continuation round: using global constraints")
        
        save_dir = f'data_new_simplified/{self.dataset}/images_poisoned/'
        os.makedirs(save_dir, exist_ok=True)
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        original_image_path = products[target_product_idx]['image_path']
        original_filename = os.path.basename(original_image_path)
        backup_adversarial_filename = f"adversarial_{original_filename}"
        adversarial_filename = f"adversarial_{original_filename.replace('.jpg', '.png')}"
        save_path = os.path.join(save_dir, adversarial_filename)
        backup_save_path = os.path.join(save_dir, backup_adversarial_filename)
        
        product_words = products[target_product_idx]['Name'].split()
        first_three_words = ' '.join(product_words[:4])
        target_text = f"1. **{first_three_words}"
        print(f"📊 Target text for image attack: {target_text}")
        
        user_content = self._create_baseline_prompt(products, image_paths)
        
        messages = [
            {
                "role": "system",
                "content": ASSSISTANT_PROMPT
            },
            {"role": "user", "content": user_content}
        ]
        
        image_inputs, _ = process_vision_info(messages)
        
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        # print(f"\n🔍 DEBUG: Final prompt structure:")
        # print("=" * 80)
        # print(prompt)
        # print("=" * 80)
        target_tokens = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)
        
        base_tensor_images = []
        adversarial_tensor = None
        original_tensor = None
        saved_original_tensor = None
        
        for i, pil_img in enumerate(image_inputs):
            if i == target_product_idx:
                adversarial_tensor = to_tensor(pil_img).to(self.device).requires_grad_(True)
                if is_first_round:
                    original_tensor = adversarial_tensor.clone().detach()
                    saved_original_tensor = original_tensor.clone()
                else:
                    original_tensor = global_original_tensor
                base_tensor_images.append(None)
            else:
                base_tensor_images.append(to_tensor(pil_img).to(self.device).detach())

        self.model.eval()

        saved_background_mask = None
        if background_mask is None:
            print(f"\n🎭 Generating spatial mask...")
            target_image_path = image_paths[target_product_idx]
            mask_array = generate_image_mask(target_image_path)
            
            if mask_array is not None:
                mask_tensor = torch.from_numpy(mask_array).float().to(self.device)
                background_mask = torch.where(mask_tensor == 0, 
                                            torch.tensor(1.0, device=self.device), 
                                            torch.tensor(object_perturbation_ratio, device=self.device))
                background_mask = background_mask.unsqueeze(0).expand(3, -1, -1)
                saved_background_mask = background_mask.clone()
                print(f"✅ Spatial mask generated")
            else:
                background_mask = None
                print("⚠️ Mask generation failed, using uniform epsilon")
        else:
            print(f"✅ Using pre-generated mask from first round")

        best_loss = float('inf')
        best_adversarial_tensor = None

        for step in range(num_steps):
            if step == num_steps // 2:
                alpha *= 0.5
                print(f"🔍 Reduced alpha to {alpha}")
            adversarial_tensor.grad = None

            tensor_image_inputs = []
            for i in range(len(image_inputs)):
                if i == target_product_idx:
                    tensor_image_inputs.append(adversarial_tensor)
                else:
                    tensor_image_inputs.append(base_tensor_images[i])

            step_vision_inputs = self.processor(
                text=[prompt],
                images=tensor_image_inputs,
                return_tensors="pt"
            ).to(self.device)

            prompt_input_ids = step_vision_inputs['input_ids']
            step_pixel_values = step_vision_inputs['pixel_values']
            step_image_grid_thw = step_vision_inputs.get('image_grid_thw')
            
            
            embedding_layer = self.model.vlm_model.get_input_embeddings()
            prompt_embeddings = embedding_layer(prompt_input_ids)
            target_embeddings = embedding_layer(target_tokens)
            sequence_embeddings = torch.cat([prompt_embeddings, target_embeddings], dim=1)
            
            full_attention_mask = torch.ones(
                sequence_embeddings.shape[0],
                sequence_embeddings.shape[1],
                dtype=torch.long,
                device=self.device
            )
            
            with torch.autocast(device_type=self.device.type, dtype=eval(f"torch.bfloat{precision}")):
                outputs = self.model.vlm_model(
                    inputs_embeds=sequence_embeddings,
                    pixel_values=step_pixel_values,
                    image_grid_thw=step_image_grid_thw,
                    attention_mask=full_attention_mask,
                    use_cache=False
                )
            
            prompt_length = prompt_input_ids.shape[1]
            target_length = target_tokens.shape[1]
            target_logits = outputs.logits[:, prompt_length-1:prompt_length+target_length-1, :]
            
            if step == 0 or step == num_steps - 1:
                with torch.no_grad():
                    target_probs = torch.softmax(target_logits, dim=-1)
                    target_token_probs = torch.gather(target_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                    
                    target_tokens_list = target_tokens[0].tolist()
                    target_probs_list = target_token_probs[0].tolist()
                    
                    print(f"\n🎯 Target Token Analysis (Iter {step}):")
                    print(f"Target string: {target_text}")
                    print("Token-wise prediction probabilities:")
                    
                    for pos, (token_id, prob) in enumerate(zip(target_tokens_list, target_probs_list)):
                        token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
                        print(f"  Position {pos}: '{token_text}' (ID: {token_id}) -> Probability: {prob:.4f}")
                    
                    print("-" * 60)
            
            target_loss = nn.CrossEntropyLoss()(
                target_logits.reshape(-1, target_logits.size(-1)),
                target_tokens.view(-1)
            )
            
            delta = adversarial_tensor - original_tensor
            loss = target_loss
            smoothness_loss_value = 0.0
            if smoothness > 0:
                smoothness_reg = self._tv_loss_perturbation(delta)
                smoothness_loss = smoothness * smoothness_reg
                smoothness_loss_value = smoothness_loss.item()
                loss = loss + smoothness_loss

            
            magnitude_loss_value = 0.0
            if magnitude > 0:
                magnitude_reg = self._magnitude_loss_perturbation(delta, background_mask, object_magnitude_weight)
                magnitude_loss = magnitude * magnitude_reg
                magnitude_loss_value = magnitude_loss.item()
                loss = loss + magnitude_loss
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_adversarial_tensor = adversarial_tensor.clone().detach()
            
            loss.backward()

            if adversarial_tensor.grad is not None:
                with torch.no_grad():
                    grad_sign = adversarial_tensor.grad.sign()
                    adversarial_tensor -= alpha * grad_sign

                    delta = adversarial_tensor - original_tensor
                    
                    if background_mask is not None:
                        epsilon_map = epsilon * background_mask
                        delta_clamped = torch.clamp(delta, -epsilon_map, epsilon_map)
                        adversarial_tensor.data = torch.clamp(original_tensor + delta_clamped, 0.0, 1.0)
                    else:
                        delta_clamped = torch.clamp(delta, -epsilon, epsilon)
                        adversarial_tensor.data = torch.clamp(original_tensor + delta_clamped, 0.0, 1.0)

                if step % 10 == 0 or step + 1 == num_steps:
                    gn = adversarial_tensor.grad.norm().item()
                    print(f"   Step {step+1}/{num_steps} | total_loss={loss.item():.4f} | target_loss={target_loss.item():.4f} | smoothness_loss={smoothness_loss_value:.4f} | magnitude_loss={magnitude_loss_value:.4f} | ‖grad‖={gn:.6f}")
            else:
                print(f"   Step {step+1}/{num_steps} | total_loss={loss.item():.4f} | target_loss={target_loss.item():.4f} | smoothness_loss={smoothness_loss_value:.4f} | magnitude_loss={magnitude_loss_value:.4f} | ⚠️ No gradient!")

        #final_adversarial_image = to_pil(adversarial_tensor.detach().cpu())
        final_adversarial_image = to_pil(best_adversarial_tensor.detach().cpu())
        final_adversarial_image.save(save_path)
        final_adversarial_image.save(backup_save_path)
        print(f"✅ Adversarial image saved: {save_path}")
        
        return_original = saved_original_tensor if is_first_round else None
        return_mask = saved_background_mask if is_first_round else None

        
        return save_path, return_original, return_mask
    
    def test_multimodal_ranking(self, num_products=4, mode='baseline', target_product_idx=None):
        """Test multimodal ranking with random products"""
        if len(self.product_list) < num_products:
            raise RuntimeError(f"❌ Need {num_products} products, only have {len(self.product_list)}")
        
        # Randomly select products
        selected_products = random.sample(self.product_list, num_products)
        
        decoded_prompt = None
        if mode in ['attack_image', 'attack_text', 'attack_multimodal']:
            if target_product_idx is None:
                target_product_idx = random.randint(0, num_products - 1)
            print(f"🎯 Target product: {selected_products[target_product_idx]['Name']}")
        
        print(f"\n🎯 Testing multimodal ranking")
        print(f"📊 Number of products: {num_products}")
        print(f"🔄 Mode: {mode}")
        print(f"🎲 Selected products:")
        for i, product in enumerate(selected_products, 1):
            marker = "🎯" if mode in ['attack_image', 'attack_text', 'attack_multimodal'] and i-1 == target_product_idx else "  "
            print(f"   {marker} {i}. {product['Name']}")
        
        # Process images - get image paths
        image_paths = self._process_images(selected_products)
        
        # Set fixed images for the wrapper (required for attack compatibility)
        print("🔧 Setting fixed images for wrapper...")
        self.model.set_fixed_images(image_paths)
        
        # For attack_text, attack_image, and attack_multimodal modes, run both baseline and attack tests
        if mode in ['attack_text', 'attack_image', 'attack_multimodal']:
            print(f"\n" + "="*60)
            print(f"🔵 BASELINE TEST (Before Attack)")
            print(f"="*60)
            
            # First run: baseline without attack
            self._run_single_ranking_test(
                selected_products, image_paths, target_product_idx
            )
            
            if mode == 'attack_multimodal':
                # Multi-round iterative attack: text → image → text → image → ...
                multimodal_params = get_multimodal_attack_params()
                NUM_ROUNDS = multimodal_params['num_rounds']
                LR_SCHEDULE = multimodal_params['lr_schedule']
                ALPHA_SCHEDULE = multimodal_params['alpha_schedule']  
                
                # Initialize for multi-round optimization
                accumulated_prompt_logits = None  
                decoded_prompt_logits = None  
                global_original_tensor = None     
                global_background_mask = None     
                current_image_paths = image_paths.copy()  
                
                print(f"\n{'='*80}")
                print(f"🔄 MULTI-ROUND MULTIMODAL ATTACK ({NUM_ROUNDS} rounds)")
                print(f"{'='*80}")
                
                for round_idx in range(NUM_ROUNDS):
                    print(f"\n{'='*80}")
                    print(f"🔄 ROUND {round_idx + 1}/{NUM_ROUNDS}")
                    print(f"{'='*80}")
                    
                    # Step 1: Text Attack
                    print(f"\n{'='*60}")
                    print(f"🟡 TEXT ATTACK (Round {round_idx + 1})")
                    print(f"{'='*60}")
                    
                    # Get learning rate for this round
                    lr_to_use = LR_SCHEDULE[round_idx] if round_idx < len(LR_SCHEDULE) else None
                    
                    decoded_prompt, selected_ids, last_y_logits = self._generate_adversarial_text(
                        selected_products, 
                        target_product_idx,
                        initial_prompt_logits=accumulated_prompt_logits, 
                        lr=lr_to_use  
                    )
                    
                    if not decoded_prompt:
                        print(f"⚠️  Failed to generate adversarial text in round {round_idx + 1}")
                        return None
                    
                    
                    # token_ids = self.processor.tokenizer(
                    #     decoded_prompt, 
                    #     return_tensors="pt", 
                    #     add_special_tokens=False
                    # )["input_ids"].to(self.device)  
                    
                    # prompt_length = token_ids.shape[1]
            
                   
                    # vocab_size = self.model.get_input_embeddings().weight.shape[0] 
                    
                    # decoded_prompt_logits = torch.full(
                    #     (1, prompt_length, vocab_size), 
                    #     -10.0, 
                    #     device=self.device, 
                    #     dtype=torch.float32
                    # )
                    # for pos in range(prompt_length):
                    #     token_id = token_ids[0, pos].item()
                    #     decoded_prompt_logits[0, pos, token_id] = 10.0
                    
                    # print(f"📝 Converted decoded_prompt to logits (length: {prompt_length})")

                    accumulated_prompt_logits = last_y_logits
                    
                    # Create products with adversarial text suffix
                    attack_products = [p.copy() for p in selected_products]
                    original_desc = attack_products[target_product_idx].get('Description', 'No description available.')
                    attack_products[target_product_idx]['Description'] = f"{original_desc}{decoded_prompt}"
                    print(f"🔍 Decoded prompt: {decoded_prompt}")
                    print(f"✅ Inserted adversarial text into target product description")
                    
                    # Step 2: Image Attack
                    print(f"\n{'='*60}")
                    print(f"🟡 IMAGE ATTACK (Round {round_idx + 1})")
                    print(f"{'='*60}")
                    
                    alpha_to_use = ALPHA_SCHEDULE[round_idx] if round_idx < len(ALPHA_SCHEDULE) else None
                    
                    poisoned_image_path, first_round_original, first_round_mask = self._generate_adversarial_image(
                        attack_products,
                        current_image_paths,
                        target_product_idx,
                        global_original_tensor=global_original_tensor,
                        background_mask=global_background_mask,
                        alpha=alpha_to_use
                    )
                    
                    # Save global constraints from first round
                    if round_idx == 0:
                        global_original_tensor = first_round_original
                        global_background_mask = first_round_mask
                        print(f"✅ Saved original tensor and mask for future rounds")
                    
                    # Update image paths (cumulative effect)
                    current_image_paths[target_product_idx] = poisoned_image_path
                    self.model.set_fixed_images(current_image_paths)


                # Final Test after all rounds
                print(f"\n{'='*80}")
                print(f"🏁 FINAL MULTIMODAL TEST (After {NUM_ROUNDS} Rounds)")
                print(f"{'='*80}")

                
                self._run_single_ranking_test(
                    attack_products,
                    current_image_paths,
                    target_product_idx
                )
                
                return None
            
            else:
                print(f"\n" + "="*60)
                if mode == 'attack_text':
                    print(f"🔴 ATTACK TEST (After Inserting Adversarial Text)")
                else:  # attack_image
                    print(f"🔴 ATTACK TEST (After Generating Adversarial Image)")
                print(f"="*60)
                
                if mode == 'attack_text':
                    # Generate adversarial text
                    decoded_prompt, _, _ = self._generate_adversarial_text(selected_products, target_product_idx)
                    if decoded_prompt:
                        # Make a copy of products for attack test (don't modify original)
                        attack_products = [p.copy() for p in selected_products]
                        
                        # Insert decoded_prompt into target product's description
                        print(f"🔍 Decoded prompt: {decoded_prompt}")
                        original_desc = attack_products[target_product_idx].get('Description', 'No description available.')
                        attack_products[target_product_idx]['Description'] = f"{original_desc}{decoded_prompt}"
                        print(f"✅ Inserted adversarial text into target product description")
                        
                        # Second run: attack with adversarial text
                        self._run_single_ranking_test(
                            attack_products, image_paths, target_product_idx
                        )
                        
                        return None
                    else:
                        print("⚠️  Failed to generate adversarial text")
                        return None
                else:
                    poisoned_image_path, _, _ = self._generate_adversarial_image(
                        selected_products, image_paths, target_product_idx
                    )
                    
                    # Update image paths
                    updated_image_paths = image_paths.copy()
                    updated_image_paths[target_product_idx] = poisoned_image_path
                    
                    # Update fixed images
                    print("🔧 Updating fixed images with adversarial image...")
                    self.model.set_fixed_images(updated_image_paths)
                    
                    # Second run: attack with adversarial image
                    self._run_single_ranking_test(
                        selected_products, updated_image_paths, target_product_idx
                    )
                    
                    return None
        else:
            # For other modes, run single test
            return self._run_single_ranking_test(selected_products, image_paths, target_product_idx)
    
    def _run_single_ranking_test(self, selected_products, image_paths, target_product_idx):
        """Run a single ranking test with given products"""
        
        content = self._create_baseline_prompt(selected_products, image_paths)
        
        messages = [
            {"role": "system", "content": ASSSISTANT_PROMPT},
            {"role": "user", "content": content}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # print(f"\n🔍 DEBUG: Final prompt structure:")
        # print("=" * 80)
        # print(text)
        # print("=" * 80)
        
        image_inputs, _ = process_vision_info(messages)
        to_tensor = transforms.ToTensor()
        tensor_image_inputs = [to_tensor(pil_img).to(self.device) for pil_img in image_inputs]
        

        inputs = self.processor(
            text=[text],
            images=tensor_image_inputs,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        print("🤖 Generating ranking...")
        with torch.no_grad():
            outputs = self.model.vlm_model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        

        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        

        print(f"\n{'='*80}")
        print(f"📊 FIRST 10 GENERATED TOKENS - PREDICTION ANALYSIS")
        print(f"{'='*80}")
        scores_tuple = outputs.scores 
        num_tokens_to_show = min(10, len(scores_tuple))
        if num_tokens_to_show > 0:
            for i in range(num_tokens_to_show):
                step_scores = scores_tuple[i][0]  # (vocab_size,)
                step_probs = torch.softmax(step_scores, dim=-1)
                
                generated_token_id = generated_ids_trimmed[0][i].item()
                generated_token_text = self.processor.tokenizer.decode([generated_token_id])
                generated_token_prob = step_probs[generated_token_id].item()
                
                top5_probs, top5_ids = torch.topk(step_probs, k=5)
                
                print(f"\n🔹 Token {i+1}: '{generated_token_text}' (ID: {generated_token_id})")
                print(f"   Probability: {generated_token_prob:.6f} ({generated_token_prob*100:.2f}%)")
                print(f"   Top-5 Predictions:")
                for rank, (prob, token_id) in enumerate(zip(top5_probs, top5_ids), 1):
                    token_text = self.processor.tokenizer.decode([token_id.item()])
                    is_selected = "✓" if token_id.item() == generated_token_id else " "
                    print(f"      {rank}. [{is_selected}] '{token_text}' - {prob.item():.6f} ({prob.item()*100:.2f}%)")
        else:
            print("⚠️  No tokens generated!")
        
        print(f"{'='*80}\n")
        
        print(f"\n🏆 Ranking Result:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        return None
    
    def run_experiments(self, num_tests=3, num_products=4, mode='baseline', target_product_idx=None):
        """Run multiple ranking experiments"""
        if not self.product_list:
            raise RuntimeError("❌ No products available!")
        
        print(f"\n{'='*60}")
        print(f"🚀 Running Multi-Model VLM Product Ranking Experiments")
        print(f"{'='*60}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Tests: {num_tests}")
        print(f"🛍️  Products per test: {num_products}")
        print(f"📂 Catalog: {self.catalog}")
        print(f"🔄 Mode: {mode}")
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            self.test_multimodal_ranking(num_products=num_products, mode=mode, target_product_idx=target_product_idx)
            print(f"✅ Test {i+1} completed successfully")
        
        print(f"\n📊 EXPERIMENT SUMMARY:")
        print(f"   Tests completed: {num_tests}")
        
        return None
    


def main():
    parser = argparse.ArgumentParser(description='Multi-Model VLM Product Reranker')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Path to model directory')
    parser.add_argument('--model_name', type=str,
                       required=True,
                       choices=['qwen2.5-vl', 'llama-3.2-vision', 'gemma-3-12b-it'],
                       help='Model type: qwen2.5-vl, llama-3.2-vision, or gemma-3-12b-it')
    parser.add_argument('--catalog', type=str, default='baby_strollers',
                       help='Product catalog')
    parser.add_argument('--dataset', type=str, default='amazon',
                       help='Dataset name')
    parser.add_argument('--num_tests', type=int, default=1,
                       help='Number of tests to run')
    parser.add_argument('--num_products', type=int, default=4,
                       help='Number of products per test')
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'attack_image', 'attack_text', 'attack_multimodal'],
                       help='Test mode: baseline, attack_image, attack_text, or attack_multimodal')
    parser.add_argument('--target_product_idx', type=int, default=None,
                       help='Target product index for attack modes')
    parser.add_argument('--seed', type=int, default=24,
                       help='Random seed for reproducibility (default: 24)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    print(f"🚀 Multi-Model VLM Product Reranker")
    print(f"📁 Model: {args.model_path}")
    print(f"🤖 Model Type: {args.model_name}")
    print(f"📊 Catalog: {args.catalog}")
    print(f"🧪 Tests: {args.num_tests}")
    print(f"🛍️  Products per test: {args.num_products}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"🎲 Target product index: {args.target_product_idx}")
    
    # Initialize reranker
    reranker = MultimodalProductReranker(
        model_path=args.model_path,
        model_name=args.model_name,
        catalog=args.catalog,
        dataset=args.dataset
    )
    
    # Run experiments
    reranker.run_experiments(
        num_tests=args.num_tests,
        num_products=args.num_products,
        mode=args.mode,
        target_product_idx=args.target_product_idx
    )
    
    print(f"\n✅ All experiments completed!")

if __name__ == "__main__":
    main()