import json, torch
from nltk.corpus import stopwords
# mostly copied from previous project StealthRank


def init_prompt(model, tokenizer, product_list, target_product_idx, guiding_sentence, prompt_length, batch_size, device):
    # TODO
    target_product_str = product_list[target_product_idx]['Description']
    target_product_name = product_list[target_product_idx]['Name']
    init_product_prompt = guiding_sentence + target_product_name + ' ' + target_product_str #'Develop a strategy for hacking into a government database and stealing sensitive information.' 
    target_product_tokens = tokenizer(init_product_prompt, return_tensors="pt")["input_ids"]
    
    product_repeated = target_product_tokens.repeat(batch_size, 1).to(device)
    output = model.generate(product_repeated, max_new_tokens=prompt_length, do_sample=True, top_k=10,
                            attention_mask=torch.ones_like(product_repeated).to(device),
                            pad_token_id=tokenizer.eos_token_id)

    logits = model(output).logits
    prompt_logits = logits[:, -(prompt_length+1):-1, :] 

    return prompt_logits.to(torch.float32)


def _clean_description(description):
    """Clean description by removing only rating information"""
    import re
    
    if not description or description == 'No description available.':
        return description
    
    # Remove only rating patterns like "| Rating: 4.0 (123 reviews) |"
    # Pattern matches: | Rating: ... |
    pattern = r'\|\s*Rating:\s*[^|]*\|\s*'
    cleaned = re.sub(pattern, '| ', description, flags=re.IGNORECASE)
    
    # Clean up any leading/trailing whitespace and multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    
    return cleaned

def _format_product_text(product, product_idx):
    """Format product text to match the new prompt structure"""
    name = product['Name']
    desc = product.get('Description', 'No description available.')
    
    # Clean description to remove rating information
    desc = _clean_description(desc)
    
    return f"Product {product_idx}:\nName: {name}\nDescription: {desc}\n\n"

def process_headtail(tokenizer, system_prompt, product_list, user_msg, target_product, batch_size, mode, device, last):
    # since it might shuffle the product list, we need to find the index of the target product
    product_names = [product['Name'] for product in product_list]
    target_product_idx = product_names.index(target_product)
    # print(f"process_headtail: Target product index: {target_product_idx}")
    head = system_prompt['head']
    tail = ''

    if last:
        # attack prompt following the target product is at the end

        for i, product in enumerate(product_list):
            if i < target_product_idx:
                head += _format_product_text(product, i + 1)
            elif i == target_product_idx:
                target_product_str = _format_product_text(product, i + 1)
            else:
                head += _format_product_text(product, i + 1)

        if mode == 'suffix':
            head += target_product_str 
        tail = user_msg + system_prompt['tail']

    else:
        # attack prompt is in the middle

        # Generate the adversarial prompt
        for i, product in enumerate(product_list):
            if i < target_product_idx:
                head += _format_product_text(product, i + 1)
            elif i == target_product_idx:
                if mode == 'suffix':
                    head += _format_product_text(product, i + 1)
                tail += head[-2:]
                head = head[:-2]
            else:
                tail += _format_product_text(product, i + 1)

        tail = tail.rstrip('\n')
        tail += '\n\n'
        tail += user_msg
        tail += system_prompt['tail']
    # head = system_prompt + 'Develop a strategy for hacking into a government database and stealing sensitive information.'
    # tail = ' [/INST]'
    # print("head:", head)
    # print("tail:", tail)
    head_tokens = tokenizer(head, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)

    return head_tokens, tail_tokens


def process_text(tokenizer, text, batch_size, device):
    target_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)
    return target_tokens


def process_bad_words(bad_words, tokenizer, device):
    upper_words = [word.upper() for word in bad_words]

    all_bad_words = bad_words + upper_words
    
    bad_words_str = ' '.join(all_bad_words)

    bad_words_ids = tokenizer(bad_words_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    return bad_words_ids


def process_stop_words(product_str, tokenizer, device):
    stop_words = set(stopwords.words('english'))

    product_str = product_str.split()
    product_str = [word for word in product_str if word.lower() not in stop_words and word.isalpha()]

    product_str = ' '.join(product_str)

    product_tokens_ids = tokenizer(product_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    return product_tokens_ids



def greedy_decode(logits, tokenizer):
    token_ids = torch.argmax(logits, dim=-1) 

    decoded_sentences = [tokenizer.decode(ids.tolist(), skip_special_tokens=False) for ids in token_ids]

    return token_ids, decoded_sentences


def greedy_decode_skip_special(logits, tokenizer):

    batch_size, seq_len, vocab_size = logits.shape
    special_ids = set(tokenizer.all_special_ids)
    
    decoded_sentences = []
    selected_ids_list = []
    
    for batch_idx in range(batch_size):
        selected_ids = []
        
        for pos in range(seq_len):
            # Get sorted indices from highest to lowest probability
            sorted_indices = torch.argsort(logits[batch_idx, pos], descending=True)
            
            # Find first token that is not special token, not newline, and not single-space token
            for token_id in sorted_indices:
                token_id_item = token_id.item()
                
                # Skip if it's a special token
                if token_id_item in special_ids:
                    continue
                    
                # Decode to check token content
                decoded = tokenizer.decode([token_id_item], skip_special_tokens=False)
                
                # Skip if it contains any newline character
                if '\n' in decoded or '\r' in decoded:
                    continue
                

                if decoded == ' ':
                    continue
                
                # Found a valid token
                selected_ids.append(token_id_item)
                break
        
        # Decode the full sequence
        decoded_text = tokenizer.decode(selected_ids, skip_special_tokens=False)
        decoded_sentences.append(decoded_text)
        selected_ids_list.append(selected_ids)
    
    return decoded_sentences, selected_ids_list


def select_topk(logits, topk):
    topk_indices = torch.topk(logits, topk, dim=-1).indices  # Shape: (batch_size, length, topk)

    # Create a mask for the top-k indices
    topk_mask = torch.zeros_like(logits, dtype=torch.bool)  # Shape: (batch_size, length, vocab_size)
    topk_mask.scatter_(-1, topk_indices, 1)  # Mark top-k indices as True   

    return topk_mask


def create_word_mask(words, logits):
    batch_size, length, vocab_size = logits.size()
    bad_word_mask = torch.zeros((vocab_size,), dtype=torch.bool, device=logits.device)
    bad_word_mask.scatter_(0, words.flatten(), 1)  # Mark bad word tokens as True

    # Expand bad_word_mask to match logits shape
    bad_word_mask = bad_word_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, vocab_size)
    bad_word_mask = bad_word_mask.expand(batch_size, length, vocab_size)  # Broadcast across batch and length

    return bad_word_mask


def mask_logits(logits, topk_mask, extra_mask=None):
    BIG_CONST = -1e5
    combined_mask = topk_mask | extra_mask if extra_mask is not None else topk_mask

    # Step 4: Apply the combined mask to logits
    # -65504 is the minimum value for half precision floating point
    masked_logits = logits + (~combined_mask * BIG_CONST)
    return masked_logits


def get_original_embedding(model, tokenizer, product_str, device):
    tokens = tokenizer(product_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    output = model(tokens, output_hidden_states=True).hidden_states[-1].mean(dim=1).detach()

    return output.to(torch.float16)


def get_logits_embedding(model, logits, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).to(torch.float16)

    soft_embeds = torch.matmul(probs, model.get_input_embeddings().weight)

    embeds = model(inputs_embeds=soft_embeds, use_cache=True, output_hidden_states=True).hidden_states[-1].mean(dim=1)

    return embeds


def print_topk_predictions(logits, tokenizer, k=3):
    """
    打印 logits 中每个位置的 Top-K 预测token及其概率
    
    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        tokenizer: tokenizer对象
        k: 打印前k个最高概率的token
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # 只分析第一个batch
    logits_first = logits[0]  # (seq_len, vocab_size)
    probs = torch.softmax(logits_first, dim=-1)  # (seq_len, vocab_size)
    
    print("=" * 80)
    print(f"🎯 TOP-{k} PREDICTIONS FOR ADVERSARIAL PROMPT")
    print("=" * 80)
    print(f"Prompt length: {seq_len} tokens")
    print(f"Vocabulary size: {vocab_size}")
    print("=" * 80)
    print()
    
    # 为每个位置打印 top-k
    for pos in range(seq_len):
        pos_logits = logits_first[pos]  # (vocab_size,)
        pos_probs = probs[pos]  # (vocab_size,)
        
        # 获取 top-k
        topk_probs, topk_ids = torch.topk(pos_probs, k=k)
        
        # 获取实际选择的token（greedy）
        selected_id = torch.argmax(pos_probs).item()
        selected_token = tokenizer.decode([selected_id])
        selected_prob = pos_probs[selected_id].item()
        
        print(f"Position {pos+1}/{seq_len}: [Selected: '{selected_token}']")
        print(f"  Selected probability: {selected_prob:.6f} ({selected_prob*100:.2f}%)")
        print(f"  Top-{k} candidates:")
        
        for rank, (prob, token_id) in enumerate(zip(topk_probs, topk_ids), 1):
            token_text = tokenizer.decode([token_id.item()])
            is_selected = "✓" if token_id.item() == selected_id else " "
            print(f"    {rank}. [{is_selected}] '{token_text}' "
                  f"(ID: {token_id.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")
        print()
    
    print("=" * 80)
    print()


