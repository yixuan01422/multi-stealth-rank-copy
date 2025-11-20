import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from process import process_headtail, greedy_decode


def get_embedding_matrix(model):
    """Get the embedding matrix from the model"""
    return model.get_input_embeddings().weight


def get_embeddings(model, input_ids):
    """Get embeddings for input ids"""
    return model.get_input_embeddings()(input_ids)


def get_banned_tokens(tokenizer):
    """Get a comprehensive list of tokens to ban for better readability"""
    banned = set(tokenizer.all_special_ids)

    allowed_punctuation = {'.', ',', '?', '!', ';', ':', '-', '(', ')'}
    
    # Add common special token IDs
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        banned.add(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        banned.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        banned.add(tokenizer.pad_token_id)
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        banned.add(tokenizer.unk_token_id)
    
    # Sample check for problematic tokens (limit to avoid full vocab scan)
    vocab_size = min(tokenizer.vocab_size, 20000)
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            stripped_token_str = token_str.strip()
            # Ban obviously problematic tokens
            if (len(stripped_token_str) == 0 or
                stripped_token_str in ['<', '>', '{', '}', '[', ']', '(', ')', '|', '\\'] or  # symbols
                all(c in '0123456789' for c in stripped_token_str) and len(stripped_token_str) > 0 or  # pure numbers
                # Ban obvious subword tokens that don't start sentences well
                (stripped_token_str.startswith(('ing', 'ed', 'er', 'est', 'ly', 'ion', 'tion', 'ment', 'ness', 'ful', 'less')) and len(stripped_token_str) < 6) or
                # Ban tokens that are just suffixes
                stripped_token_str in ['ists', 'ement', 'arium', 'ship', 'ness', 'tion', 'sion'] or
                # Ban very short tokens that are likely fragments
                (len(stripped_token_str) <= 2 and not stripped_token_str.isalpha() and stripped_token_str not in allowed_punctuation)):
                banned.add(token_id)
        except:
            banned.add(token_id)
    
    return list(banned)

def get_top3_token_sequences(model, tokenizer, head_tokens, fixed_control_sequence, 
                             current_optimizing_token, tail_tokens, num_tokens=3):
    base_input_ids = torch.cat([
        head_tokens[0],
        fixed_control_sequence,
        current_optimizing_token,
        tail_tokens[0]
    ], dim=0).unsqueeze(0)
    
    candidates = [(base_input_ids, 0.0, "")]
    
    print(f"Starting sequence generation for {num_tokens} tokens...")
    print("="*80)
    
    for step in range(num_tokens):
        print(f"Step {step + 1}: Generating token {step + 1}")
        new_candidates = []
        
        for seq_input_ids, current_log_prob, current_text in candidates:
            with torch.no_grad():
                next_token_logits = model(input_ids=seq_input_ids).logits[:, -1, :]
                next_token_probs = next_token_logits.softmax(dim=-1).squeeze(0)
                
                top_k = 5 if step == 0 else 3
                top_tokens_probs, top_tokens = next_token_probs.topk(top_k)
                
                for token_id, token_prob in zip(top_tokens, top_tokens_probs):
                    new_input_ids = torch.cat([
                        seq_input_ids, 
                        token_id.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    new_log_prob = current_log_prob + torch.log(token_prob).item()
                    
                    token_text = tokenizer.decode(token_id, skip_special_tokens=True)
                    new_text = current_text + token_text
                    
                    new_candidates.append((new_input_ids, new_log_prob, new_text))
                    
                    # print(f"  Candidate: '{current_text}' + '{token_text}' = '{new_text}' "
                        #   f"(step_prob: {token_prob.item():.4f}, cumul_log_prob: {new_log_prob:.4f})")
        
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:3]
        
        print(f"  Top 3 after step {step + 1}:")
        for i, (_, log_prob, text) in enumerate(candidates):
            prob = torch.exp(torch.tensor(log_prob)).item()
            print(f"    {i+1}. '{text}' (cumulative_prob: {prob:.6f})")
        print("-" * 60)
    
    print("FINAL TOP-3 TOKEN SEQUENCES:")
    print("="*80)
    final_results = []
    
    for i, (final_input_ids, final_log_prob, final_text) in enumerate(candidates):
        final_prob = torch.exp(torch.tensor(final_log_prob)).item()
        final_results.append((final_text, final_prob, final_input_ids))
        
        print(f"{i+1}. Sequence: '{final_text}'")
        print(f"   Probability: {final_prob:.8f}")
        print(f"   Log Probability: {final_log_prob:.4f}")
        
        generated_tokens = final_input_ids[0, base_input_ids.shape[1]:]
        print(f"   Generated token IDs: {generated_tokens.tolist()}")
        print(f"   Individual tokens: {[tokenizer.decode(tid) for tid in generated_tokens]}")
        print("-" * 60)
    
    return final_results

def autodan_attack_control(model, tokenizer, system_prompt, user_msg,
                          max_length, target_token_lists, product_list, 
                          target_product, target_product_tokens, logger, table, **kwargs):
    device = model.device
    mode = kwargs['mode']
    search_mode = kwargs.get('search_mode', 'optimize')
    batch_size = kwargs.get('batch_size', 512)
    topk = kwargs.get('topk', 256)
    w_tar_1 = kwargs.get('w_tar_1', 10.0)
    w_tar_2 = kwargs.get('w_tar_2', 10.0)
    ctrl_temp = kwargs.get('ctrl_temp', 1.0)
    
    # Beam search parameters
    beam_width = kwargs.get('beam_width', 4)  # Number of sequences to maintain
    
    # Allow switching between greedy selection and multinomial sampling
    sampling_method = kwargs.get('sampling_method', 'multinomial')
    
    # Loss type configuration
    loss_type = kwargs.get('loss_type', 'hard')
    loss_kwargs = {
        'max_positions': kwargs.get('max_positions', 20),
        'lambda_decay': kwargs.get('lambda_decay', 0.1),
        'punishment_value': kwargs.get('punishment_value', 100.0)
    }
    
    # Get target product text for fluency context
    target_product_idx = next(i for i, p in enumerate(product_list) if p['Name'] == target_product)
    target_product_text = f"{product_list[target_product_idx]['Description']}"

    embed_weights = get_embedding_matrix(model)
    actual_vocab_size = embed_weights.shape[0]
    special_ids = set(tokenizer.all_special_ids)
    special_ids = {id for id in special_ids if 0 <= id < actual_vocab_size}
    
    # Get banned tokens
    banned_tokens = get_banned_tokens(tokenizer)
    
    # State
    current_length = 0
    position_converged = True
    
    print(f"Target product: {target_product}")
    print(f"Using {len(target_token_lists)} different target templates for training")
    print(f"Loss type: {loss_type}")
    if loss_type == 'soft':
        print(f"Soft loss parameters: max_positions={loss_kwargs['max_positions']}, "
              f"lambda_decay={loss_kwargs['lambda_decay']}, "
              f"punishment_value={loss_kwargs['punishment_value']}")
    
    # Print entropy-adaptive weighting info
    if kwargs.get('use_entropy_adaptive_weighting', False):
        print(f"Entropy-adaptive weighting ENABLED:")
        print(f"  - Alpha parameter: {kwargs.get('entropy_alpha', 100.0)}")
        print(f"  - Base w_tar_2: {w_tar_2} (will be dynamically adjusted)")
        print(f"  - Formula: w_tar_2 = alpha * (H_max - H(p_read)) / H_max")
    
    if kwargs.get('position_dependent_weighting', False):
        print(f"Position-dependent weighting ENABLED:")
        w_init = kwargs.get('position_w_init', w_tar_2)
        lambda_decay_pos = kwargs.get('position_lambda_decay', 0.05)
        min_weight = kwargs.get('position_min_weight', 0.1)
        print(f"  - Initial weight w_init: {w_init}")
        print(f"  - Decay rate lambda: {lambda_decay_pos}")
        print(f"  - Minimum weight: {min_weight}")
        print(f"  - Formula: w_tar_2(t) = {w_init} * exp(-{lambda_decay_pos} * t)")
    else:
        print(f"Using fixed weights: w_tar_1={w_tar_1}, w_tar_2={w_tar_2}")
    
    n_steps = kwargs.get('n_steps', 100)
    current_sequence = torch.tensor([], dtype=torch.long, device=device)
    current_position = 0  
    history_set = set()
    current_optimizing_token = None
    fixed_control_sequence = torch.tensor([], dtype=torch.long, device=device)
    
    print(f"Starting AutoDAN position-by-position optimization")
    print(f"Total optimization steps: {n_steps}")
    print(f"Max sequence length: {max_length}")
    print(f"Target product: {target_product}")
    print(f"Using {len(target_token_lists)} different target templates")
    
    # Main optimization loop - fixed number of steps
    for step in range(1, n_steps + 1):
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION STEP {step} / {n_steps}")
        print(f"Current position: {current_position}")
        print(f"{'='*80}")

        # Check if we've reached max length
        if current_position >= max_length:
            print(f"Reached maximum sequence length ({max_length}), stopping optimization")
            break
        
        # Initialize new position with random token if needed
        if position_converged:
            position_converged = False
            
            vocab_size = embed_weights.shape[0]
            valid_token_ids = [i for i in range(vocab_size) if i not in special_ids]
            if len(valid_token_ids) == 0:
                print("Warning: No valid tokens available for initialization!")
                break
            
            # Initialize new position with random token
            random_token_id = random.choice(valid_token_ids)
            random_token = torch.tensor([random_token_id], dtype=torch.long, device=device)
            current_optimizing_token = random_token 
            
            # Extend sequence with random token
            print(f"Initialized position {current_position} with random token {random_token_id}: {tokenizer.decode([random_token_id], skip_special_tokens=True)}")
            
            # Reset history set for new position
            history_set = set()

        print(f"Fixed control sequence: {tokenizer.decode(fixed_control_sequence, skip_special_tokens=True)}")
        print(f"Optimizing token at position {current_position}: {tokenizer.decode(current_optimizing_token, skip_special_tokens=True)}")
        
        # Prepare head and tail tokens
        if kwargs.get('random_order', False):
            random.shuffle(product_list)

        # TODO: random order construction
        head_tokens, tail_tokens = process_headtail(
            tokenizer, system_prompt, product_list, user_msg,
            target_product, 1, mode, device, last=False
        )

        # Output product probability distribution
        # input_ids = torch.cat([
        #         head_tokens[0],
        #         fixed_control_sequence,
        #         current_optimizing_token,
        #         tail_tokens[0]
        #     ], dim=0).unsqueeze(0)
        # next_token_logits = model(input_ids=input_ids).logits[:, -1, :]
        # top_5_tokens = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).indices
        # top_5_tokens_probs = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).values
        # print("top 5 tokens:")
        # for i in range(5):
        #     print(f" {i+1}. token: {tokenizer.decode(top_5_tokens[i])} with probability {top_5_tokens_probs[i].item():.4f}")
        # print("="*80)
        # top_3_tokens = get_top3_token_sequences(model, tokenizer, head_tokens, fixed_control_sequence, 
        #                                         current_optimizing_token, tail_tokens, num_tokens=3)
        
        # TODO: randomly select target template
        if len(target_token_lists) == 1:
            selected_target_idx = 0
        else:
            selected_target_idx = random.randint(0, len(target_token_lists) - 1)
        selected_target_tokens = target_token_lists[selected_target_idx]
        
        # Step 1: Preliminary selection to get gradients and readability scores
        target_grad, last_tok_ll = preliminary_select(
            model, tokenizer, head_tokens, fixed_control_sequence, current_optimizing_token,
            tail_tokens, selected_target_tokens, target_product_text,
            loss_type=loss_type
        )

        # Step 2: Use existing token sampling method to get candidate set for this position
        candidate_ids = sample_control_tokens(
            target_grad, last_tok_ll, batch_size, topk,
            tokenizer, banned_tokens, w_tar_1, temperature=ctrl_temp, debug=True
        )

        if len(candidate_ids) == 0:
            print("Warning: No valid candidates found!")
            break
        
        # Step 3: Check if current token x is in candidate set X (for convergence)
        if current_optimizing_token not in candidate_ids:
            # Greedily add x to candidate set to ensure convergence
            candidate_ids = torch.cat([candidate_ids, current_optimizing_token])
            print(f"  Added current token {current_optimizing_token.item()} to ensure convergence")
        
        # Step 4: Create candidate sequences by appending the current token to the fixed control sequence
        candidate_sequences = []
        for candidate_token in candidate_ids:
            # Create sequence by appending the current token to the fixed control sequence
            candidate_seq = torch.cat([fixed_control_sequence, candidate_token.unsqueeze(0)])
            candidate_sequences.append(candidate_seq)
        candidate_sequences = torch.stack(candidate_sequences)
        
        # Step 5: Evaluate all candidates using existing fine_select method
        # For ablation
        loss_target, loss_control = fine_select(
            model, tokenizer, head_tokens, candidate_sequences,
            tail_tokens, selected_target_tokens, last_tok_ll,
            loss_type=loss_type, debug=True, w_tar_2=0.0, **loss_kwargs
        )
        
        # Step 6: Combine objectives for final selection
        # Entropy-adaptive weighting for fine selection
        entropy = None
        max_entropy = None
        adaptive_w_ctrl_2 = None
        # For ablation
        w_ctrl_2 = 0.0
        
        # Loss function
        if kwargs.get('use_entropy_adaptive_weighting', False):
            # Calculate entropy of the readability distribution
            readability_probs = torch.exp(last_tok_ll[0])  # Shape: [vocab_size]
            
            # Normalize probabilities to ensure they sum to 1 (numerical stability)
            prob_sum = readability_probs.sum()
            if prob_sum > 1e-20:
                readability_probs = readability_probs / prob_sum
            else:
                readability_probs = torch.ones_like(readability_probs) / readability_probs.shape[0]
            
            # Calculate maximum entropy H_max = log(|V|)
            vocab_size_float = last_tok_ll.shape[1]
            max_entropy = torch.log(torch.tensor(vocab_size_float, dtype=torch.float32, device=device))
            
            # Calculate entropy H(p_read) = -sum(p * log(p))
            epsilon = 1e-15
            valid_mask = readability_probs > epsilon
            if valid_mask.sum() == 0:
                entropy = max_entropy
            else:
                valid_probs = readability_probs[valid_mask]
                log_valid_probs = torch.log(valid_probs)
                entropy = -torch.sum(valid_probs * log_valid_probs)
            
            # Safety checks for entropy
            if torch.isnan(entropy) or torch.isinf(entropy):
                print(f"Warning: Invalid entropy {entropy:.4f}, falling back to fixed weight")
                adaptive_w_ctrl_2 = torch.tensor(1.0, device=device)
            else:
                # Calculate adaptive weight
                alpha = kwargs.get('entropy_alpha', 10.0)
                entropy_ratio = torch.clamp(entropy / max_entropy, 0.0, 1.0)
                adaptive_w_ctrl_2 = alpha * (1.0 - entropy_ratio)
                adaptive_w_ctrl_2 = torch.clamp(adaptive_w_ctrl_2, min=0.0)
            
            w_ctrl_2 = adaptive_w_ctrl_2
            
            print(f"    Initial Entropy: {entropy:.4f} / {max_entropy:.4f} | "
                f"Adaptive w_ctrl_2: {w_ctrl_2:.4f}")

        if kwargs.get('position_dependent_weighting', False):
            # Position-dependent weighting: w_2(t) = w_init * exp(-lambda * t)
            w_init = kwargs.get('position_w_init', w_tar_2)  # Initial weight (fallback to w_tar_2)
            lambda_decay_pos = kwargs.get('position_lambda_decay', 0.05)  # Decay rate
            
            # Current position is the length of the current optimizing sequence
            current_pos = current_position  # This is the token position in suffix
            
            # Calculate position-dependent w_tar_2
            position_w_tar_2 = w_init * torch.exp(torch.tensor(-lambda_decay_pos * current_pos, device=device))
            
            # Ensure minimum weight
            min_weight = kwargs.get('position_min_weight', 0.1)
            position_w_tar_2 = torch.clamp(position_w_tar_2, min=min_weight)
            
            w_tar_2 = position_w_tar_2
            
            print(f"    Position {current_pos}: w_tar_2 = {w_init:.4f} * exp(-{lambda_decay_pos:.3f} * {current_pos}) = {w_tar_2:.4f}")
        
        combined_losses = w_ctrl_2 * loss_control + w_tar_2 * loss_target

        # Step 7: Select optimal token using Multinomial Sampling or greedy selection
        temperature = kwargs.get('ctrl_temp', 1.0)
        if sampling_method == 'multinomial' and temperature > 0:
            # Multinomial sampling with negative losses
            probs = F.softmax(-combined_losses / temperature, dim=0)
            selected_idx = torch.multinomial(probs, 1).item()
            selected_token_id = candidate_ids[selected_idx].item()
            print(f"  Selected token {selected_token_id}: '{tokenizer.decode([selected_token_id], skip_special_tokens=True)}' via multinomial sampling (temp={temperature})")
        else:
            # Greedy selection (top-1)
            selected_idx = torch.argmin(combined_losses).item()
            selected_token_id = candidate_ids[selected_idx].item()
            print(f"  Selected token {selected_token_id}: '{tokenizer.decode([selected_token_id], skip_special_tokens=True)}' via greedy selection")
        
        # Get the selected token and sequence
        selected_token = candidate_ids[selected_idx]
        selected_sequence = candidate_sequences[selected_idx]
        selected_loss_target = loss_target[selected_idx].item()
        selected_loss_control = loss_control[selected_idx].item()
        selected_combined_loss = combined_losses[selected_idx].item()
        
        # Step 8: Convergence check for this position
        # Check if x^{(top)} ∈ H (history set)
        top_idx = torch.argmin(combined_losses).item()
        top_token = candidate_ids[top_idx]
        top_token_id = top_token.item()

        # Update the current optimizing token
        current_optimizing_token = selected_token.unsqueeze(0)
        position_converged = top_token_id in history_set
        if position_converged:
            print(f"  Position {current_position} converged! Top token {top_token_id}: '{tokenizer.decode([top_token_id], skip_special_tokens=True)}' already in history")
            fixed_control_sequence = selected_sequence.clone()
            current_position += 1
            print(f"  Moving to position {current_position}")
        else:
            history_set.add(top_token_id)
        
        current_sequence = selected_sequence.clone()
        selected_text = tokenizer.decode(selected_sequence, skip_special_tokens=True)
        
        print(f"  Current sequence: '{selected_text}'")
        print(f"  Target loss: {selected_loss_target:.4f}")
        print(f"  Control loss: {selected_loss_control:.4f}")
        print(f"  Combined loss: {selected_combined_loss:.4f}")
        print(f"  History set size for position {current_position - (1 if position_converged else 0)}: {len(history_set)}")
        
        # Log progress
        if logger:
            log_dict = {
                "train/step": step,
                "train/position": current_position - (1 if position_converged else 0),
                "train/target_loss": selected_loss_target,
                "train/control_loss": selected_loss_control,
                "train/combined_loss": selected_combined_loss,
                "train/history_set_size": len(history_set),
            }
            logger.log(log_dict, step=step)
        
        # Periodic evaluation (COMMENTED OUT - not needed for training)
        # if position_converged:
        #     print(f"\n=== EVALUATION AT STEP {step} ===")
        #     
        #     # Randomly construct product order for evaluation
        #     complete_prompts = []
        #     
        #     for _ in range(5): 
        #         # random.shuffle(product_list)
        #         head_tokens, tail_tokens = process_headtail(
        #             tokenizer, system_prompt, product_list, user_msg,
        #             target_product, 1, mode, device, last=False
        #         )
        #         complete_prompts.append(torch.cat([head_tokens[0], current_sequence, tail_tokens[0]], dim=0))
        #     
        #     table, rank = log_eval_result(model, tokenizer,
        #                      selected_text, 
        #                      torch.stack(complete_prompts, dim=0),
        #                      step, product_list, target_product, table, logger)
        #     
        #     print(f"Target product '{target_product}' rank: {rank}")
        #     print(f"=== END EVALUATION ===\n")
    
    # Final sequence length for later use
    current_length = len(current_sequence)
    
    print(f"\n{'='*60}")
    print(f"AutoDAN optimization completed after {n_steps} steps")
    print(f"Final sequence length: {current_length}")
    print(f"Final position reached: {current_position}")
    
    # Return the final optimized attack text
    if current_length > 0:
        final_text = tokenizer.decode(current_sequence, skip_special_tokens=True)
        print(f"Final optimized attack text: '{final_text}'")
        print(f"{'='*60}")
        return final_text
    else:
        print("No valid sequences found during optimization")
        print(f"{'='*60}")
        return ""
    
    # # Original evaluation code (commented out):
    # # Evaluate best control sequence
    # best_embed_weights = get_embedding_matrix(model)
    # best_actual_vocab_size = best_embed_weights.shape[0]
    # best_control_tokens_clamped = torch.clamp(fixed_control_sequence, 0, best_actual_vocab_size - 1)
    # best_control_logits = torch.zeros(1, len(best_control_tokens_clamped), best_actual_vocab_size).to(device)
    # best_control_logits.scatter_(2, best_control_tokens_clamped.unsqueeze(0).unsqueeze(2), 1)
    # best_control_logits = torch.log(best_control_logits + 1e-10)
    # 
    # # Use fresh head/tail for final evaluation
    # final_head_tokens, final_tail_tokens = process_headtail(
    #     tokenizer, system_prompt, product_list, user_msg,
    #     target_product, 1, mode, device, last=False
    # )
    # 
    # from experiment.attack import log_result
    # final_table, final_rank = log_result(
    #     model, tokenizer, final_head_tokens, best_control_logits,
    #     final_tail_tokens, target_token_lists[0], step, product_list,
    #     target_product, table, logger, topk, 1.0
    # )
    # 
    # final_control_text = tokenizer.decode(fixed_control_sequence, skip_special_tokens=True)
    # print(f"Optimization complete after {current_length} steps")
    # print(f"Best control sequence ({len(fixed_control_sequence)} tokens): '{final_control_text}'")
    # print(f"Final rank: {final_rank}")
    # 
    # # Log final evaluation metrics
    # if logger:
    #     final_metrics = {
    #         "eval/final_rank": final_rank,
    #     }
    #     logger.log(final_metrics, step=step)
    # 
    # print(f"{'='*80}")
    # 
    # return final_table, final_rank


def sample_control_tokens(target_grad, last_tok_ll, batch_size, topk, 
                         tokenizer, banned_tokens=[], 
                         w_tar_1=1.0, temperature=1.0, debug=True):
    """
    Sample control tokens based on combined objective.
    """
    device = target_grad.device
    actual_vocab_size = target_grad.shape[1]
    
    # Apply bans to target gradient
    target_grad = target_grad.clone()
    banned_ids = list(set(banned_tokens))
    target_grad[:, banned_ids] = -float('inf')
    
    # Get objectives
    target_obj = target_grad[0]
    control_obj = last_tok_ll[0]
    
    # Ensure control_obj matches target_obj size
    if control_obj.shape[0] != target_obj.shape[0]:
        # Truncate or pad to match
        min_size = min(control_obj.shape[0], target_obj.shape[0])
        target_obj = target_obj[:min_size]
        control_obj = control_obj[:min_size]
    
    # Combine objectives directly with weights
    # For ablation 
    combined_obj = control_obj + w_tar_1 * target_obj
    
    # Get top-k candidates
    valid_mask = ~torch.isinf(combined_obj)
    if valid_mask.sum() == 0:
        print("Warning: No valid tokens found!")
        return torch.tensor([], device=device), torch.tensor([], device=device)
    
    valid_scores = combined_obj[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    # Additional safety check: ensure all indices are within vocab bounds
    vocab_mask = (valid_indices >= 0) & (valid_indices < actual_vocab_size)
    valid_indices = valid_indices[vocab_mask]
    valid_scores = valid_scores[vocab_mask]
    
    if len(valid_indices) == 0:
        print("Warning: No valid tokens within vocabulary bounds!")
        return torch.tensor([], device=device), torch.tensor([], device=device)
    
    k = min(topk, len(valid_scores))
    _, topk_idx = torch.topk(valid_scores, k)
    topk_indices = valid_indices[topk_idx]

    
    if debug:
        print(f"\n=== TOKEN SELECTION DEBUG ===")
        print(f"Number of banned tokens: {len(banned_ids)}")
        print(f"Valid tokens: {valid_mask.sum()}")
        print(f"Target gradient range: [{target_obj[valid_mask].min().item():.4f}, {target_obj[valid_mask].max().item():.4f}]")
        print(f"Control range: [{control_obj[valid_mask].min().item():.4f}, {control_obj[valid_mask].max().item():.4f}]")
        print(f"Combined objective range: [{combined_obj[valid_mask].min().item():.4f}, {combined_obj[valid_mask].max().item():.4f}]")
        
        # Show top candidates
        print(f"\nTop 10 candidate tokens:")
        top_display = min(10, len(topk_indices))
        topk_tokens = [tokenizer.decode([topk_indices[i].item()]) for i in range(top_display)]
        topk_scores = [combined_obj[topk_indices[i]].item() for i in range(top_display)]
        
        for i in range(top_display):
            token_text = repr(topk_tokens[i])
            score = topk_scores[i]
            print(f"  {i+1:2d}. Token {topk_indices[i].item():5d}: {token_text} | Score: {score:.4f}")
    
    return topk_indices


def calculate_soft_position_loss(completion_logits, target_tokens, device, **loss_kwargs):
    # Get parameters from kwargs
    max_positions = loss_kwargs.get('max_positions', 20)
    lambda_decay = loss_kwargs.get('lambda_decay', 0.01)
    weight_type = loss_kwargs.get('weight_type', 'exp')

    predictions_log_softmax = F.log_softmax(completion_logits, dim=-1)
    batch_size, seq_len, _ = predictions_log_softmax.shape
    targets = target_tokens.clone()
    target_len = targets.shape[1]

    L = max_positions
    if L < target_len:
        raise ValueError(f"max_positions (L={L}) must be >= target_len (m={target_len})")

    num_positions = L - target_len + 1

    positions = torch.arange(1, num_positions + 1, device=device, dtype=torch.float32)
    if weight_type == 'exp':
        weights = torch.exp(-lambda_decay * positions)
    elif weight_type == 'inv':
        weights = 1.0 / positions
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}. Must be 'exp' or 'inv'")

    # Calculate log P_i^span(t) for all starting positions `i` in a batched way
    expanded_targets = targets.unsqueeze(2) # Shape: (batch_size, target_len, 1)

    all_target_log_probs = []

    for i in range(num_positions):
        span_log_probs = torch.gather(
            predictions_log_softmax[:, i:i+target_len, :], 2, expanded_targets
        ).squeeze(2)
        all_target_log_probs.append(span_log_probs)
    
    # Stack along the positions dimension
    # Shape results in: (num_positions, batch_size, target_len)
    # Permute to: (batch_size, num_positions, target_len)
    stacked_target_log_probs = torch.stack(all_target_log_probs, dim=0).permute(1, 0, 2)

    # Sum the log probabilities to get the log of the joint probability for each span
    span_log_probs = stacked_target_log_probs.sum(dim=2)  # Shape: (batch_size, num_positions)
    per_sequence_loss = -(span_log_probs * weights).sum(dim=1)

    return per_sequence_loss

def calculate_first_position_loss(completion_logits, target_tokens):
    predictions_log_softmax = F.log_softmax(completion_logits, dim=-1)
    target_len = target_tokens.shape[1]
    
    expanded_targets = target_tokens.unsqueeze(2)
    first_position_log_probs = torch.gather(
        predictions_log_softmax[:, :target_len, :], 
        2, 
        expanded_targets
    ).squeeze(2)
    
    first_position_joint_log_prob = first_position_log_probs.sum(dim=1)
    return -first_position_joint_log_prob

# TODO: this is not used in the current implementation
def compute_pairwise_ranking_loss(completion_logits, target_tokens, device,
                                  margin=1.0, max_positions=20, **loss_kwargs):
    """
    Computes a pairwise ranking loss that encourages the target to appear
    earlier in the sequence.

    The loss is the sum over all pairs of positions (j, k) where j < k:
    loss = max(0, margin + loss_j - loss_k)

    where loss_j is the cross-entropy of the target if it started at position j.
    This encourages loss_j to be less than loss_k.

    Args:
        completion_logits (torch.Tensor): Logits for the generation window.
                                          Shape: [1, window_len, vocab_size].
        target_tokens (torch.Tensor): The tokenized target sequence.
                                      Shape: [1, target_len].
        device (torch.device): The device to run on.
        margin (float): The margin hyperparameter for the ranking loss.
        max_positions (int): The max number of start positions to consider.

    Returns:
        torch.Tensor: The computed scalar loss.
    """
    target_seq = target_tokens[0]
    target_len = target_seq.shape[0]

    max_possible_starts = completion_logits.shape[1] - target_len + 1
    num_starts = min(max_positions, max_possible_starts)

    if num_starts <= 1:
        if max_possible_starts > 0:
            span_logits = completion_logits[:, :target_len, :]
            return F.cross_entropy(span_logits.squeeze(0), target_seq)
        return torch.tensor(0.0, device=device)

    # Calculate cross-entropy loss for each possible starting position
    positional_losses = []
    for j in range(num_starts):
        start_idx = j
        end_idx = j + target_len
        span_logits = completion_logits[:, start_idx:end_idx, :] # Shape: [1, target_len, V]
        loss_j = F.cross_entropy(span_logits.squeeze(0), target_seq)
        positional_losses.append(loss_j)

    # Compute the pairwise ranking loss
    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0
    for j in range(num_starts):
        for k in range(j + 1, num_starts):
            # We want loss_j to be smaller than loss_k for j < k
            pair_loss = F.relu(margin + positional_losses[j] - positional_losses[k])
            total_loss += pair_loss
            num_pairs += 1

    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)


def preliminary_select(model, tokenizer, 
                       head_tokens, 
                       fixed_control_tokens, 
                       optimizing_token,
                       tail_tokens, 
                       target_tokens, 
                       target_product_text, 
                       loss_type='hard'):
    device = model.device
    embed_weights = get_embedding_matrix(model)
    
    # Debug: Print control tokens being processed
    if len(fixed_control_tokens) > 0:
        control_text = tokenizer.decode(fixed_control_tokens, skip_special_tokens=True)
        print(f"[DEBUG] preliminary_select: control_tokens = '{control_text}' (length: {len(fixed_control_tokens)})")
    else:
        print(f"[DEBUG] preliminary_select: control_tokens = <empty> (length: 0)")
    
    # Position of the optimizing token
    optimizing_token_position = head_tokens.shape[1] + len(fixed_control_tokens)
    
    # Build input sequence: head + fixed_control + optimizing_token + tail
    input_ids = torch.cat([head_tokens[0], fixed_control_tokens, optimizing_token, tail_tokens[0]], dim=0)

    print("[DEBUG] preliminary_select: optimizing token:", tokenizer.decode(input_ids[optimizing_token_position], skip_special_tokens=True), " on position:", optimizing_token_position)
    
    # Create one-hot for the optimizing token position
    vocab_size = embed_weights.shape[0]
    one_hot = torch.zeros(1, vocab_size, device=device, dtype=embed_weights.dtype)

    # Initialize with the current token at that position
    one_hot.scatter_(1, optimizing_token.unsqueeze(0), torch.ones(1, 1, device=device, dtype=embed_weights.dtype))
    one_hot.requires_grad_()
    
    # Get embeddings for the optimizing token
    optimized_token_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # Get embeddings for other parts
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    
    # Replace the optimizing token embedding with our gradient-enabled one
    full_embeds = torch.cat([
        embeds[:, :optimizing_token_position, :],      # Before the optimizing token
        optimized_token_embeds,                        # The optimizing token
        embeds[:, optimizing_token_position+1:, :]     # After the optimizing token
    ], dim=1)
    
    # Add target embeddings for loss calculation
    target_embeds = get_embeddings(model, target_tokens)
    full_embeds_with_target = torch.cat([full_embeds, target_embeds], dim=1)
    
    # Forward pass
    logits = model(inputs_embeds=full_embeds_with_target).logits
    
    # Compute target loss (ranking objective)
    # The target should appear after the control sequence
    target_start = full_embeds.shape[1] - 1
    target_end = target_start + target_tokens.shape[1]
    loss = F.cross_entropy(
        logits[0, target_start:target_end, :], 
        target_tokens[0]
    )

    # if loss_type == 'soft':
    #     loss = calculate_first_position_loss(logits, target_tokens, device)
    # else:
    #     loss = F.cross_entropy(
    #         logits[0, target_start:target_end, :], 
    #         target_tokens[0]
    #     )

    loss.backward()
    
    # Check if gradients were computed
    if one_hot.grad is None:
        print(f"Warning: No gradients computed for loss_type={loss_type}, loss={loss.item():.6f}")
        target_grad = torch.zeros_like(one_hot)
    else:
        target_grad = -one_hot.grad.clone()
    
    # Compute fluency scores for the optimized token position
    with torch.no_grad():
        # Build fluency context: target product description + fixed control sequence
        control_prefix_text = tokenizer.decode(fixed_control_tokens, skip_special_tokens=True)
        fluency_context = target_product_text + control_prefix_text
        
        # Debug: Print fluency context
        print(f"[DEBUG] preliminary_select: fluency_context = '{fluency_context}'")
        
        # Tokenize fluency context
        context_tokens = tokenizer.encode(fluency_context, return_tensors='pt').to(device)
        
        # Get next token probabilities
        context_logits = model(input_ids=context_tokens).logits
        last_tok_ll = context_logits[:, -1, :].log_softmax(dim=1)
    
    return target_grad, last_tok_ll


def fine_select(model, 
                tokenizer, 
                head_tokens, candidate_sequences,
                tail_tokens, target_tokens, last_tok_ll, 
                control_loss_method='last_token_ll',
                loss_type='hard', debug=True, w_tar_2=1.0, **kwargs):
    device = model.device
    batch_size = candidate_sequences.shape[0]
    
    # Extract loss_kwargs from kwargs
    loss_kwargs = {k: v for k, v in kwargs.items() if k not in ['mini_batch_size']}
    
    # Compute losses for each candidate sequence
    loss_target = torch.zeros(batch_size).to(device)
    loss_control = torch.zeros(batch_size).to(device)
    
    if debug:
        print(f"\n=== SEQUENTIAL EVALUATION DEBUG ===")
        print(f"Evaluating {batch_size} candidate sequences with {loss_type} loss")
        print(f"Target loss type: {loss_type}")
        print(f"Control loss method: {control_loss_method}")
    
    with torch.no_grad():
        # Compute control loss
        for i in range(batch_size):
            last_token = candidate_sequences[i, -1]
            if 0 <= last_token < last_tok_ll.shape[1]:
                loss_control[i] = -last_tok_ll[0, last_token]
            else:
                loss_control[i] = 100.0 

        # Compute target loss
        if loss_type == 'soft' or loss_type == 'ranking':
            max_positions = loss_kwargs.get('max_positions', 20)
            mini_batch_size = kwargs.get('mini_batch_size', 4)  # Default mini-batch size of 4
            
            # Process in mini-batches
            for start_idx in tqdm(range(0, batch_size, mini_batch_size), desc="Evaluating candidates"):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                current_mini_batch_size = end_idx - start_idx
                
                # Build batch of prompt inputs (head + control + tail)
                prompt_input_ids_list = []
                for i in range(start_idx, end_idx):
                    current_control_sequence = candidate_sequences[i]
                    prompt_input_ids = torch.cat([
                        head_tokens[0],
                        current_control_sequence,
                        tail_tokens[0]
                    ], dim=0)
                    prompt_input_ids_list.append(prompt_input_ids)
                
                # Pad sequences to same length for batching
                max_len = max(seq.shape[0] for seq in prompt_input_ids_list)
                prompt_input_ids_batch = []
                for seq in prompt_input_ids_list:
                    if seq.shape[0] < max_len:
                        # Pad with tokenizer.pad_token_id if available, otherwise use 0
                        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
                        if pad_token_id is None:
                            pad_token_id = 0
                        padding = torch.full((max_len - seq.shape[0],), pad_token_id, 
                                           dtype=seq.dtype, device=seq.device)
                        seq = torch.cat([seq, padding], dim=0)
                    prompt_input_ids_batch.append(seq)
                
                prompt_input_ids_batch = torch.stack(prompt_input_ids_batch)  # Shape: [mini_batch_size, max_len]
                
                # Get completion logits for the entire mini-batch
                completion_logits_batch = expand_logits_batch(model, prompt_input_ids_batch, max_positions)
                
                # Calculate losses for each sequence in the mini-batch
                for i, batch_idx in enumerate(range(start_idx, end_idx)):
                    # Extract logits for this specific sequence
                    completion_logits = completion_logits_batch[i:i+1]  # Shape: [1, max_positions, vocab_size]
                    
                    if loss_type == 'soft':
                        loss_target[batch_idx] = calculate_soft_position_loss(
                            completion_logits, target_tokens, device, **loss_kwargs
                        )
                        # loss_target[batch_idx] = calculate_first_position_loss(completion_logits, target_tokens) 
                    else:
                        loss_target[batch_idx] = compute_pairwise_ranking_loss(
                            completion_logits, target_tokens, device, **loss_kwargs
                        )

        elif loss_type == 'hard':
            for i in range(batch_size):
                input_ids = torch.cat([
                    head_tokens[0], 
                    candidate_sequences[i], 
                    tail_tokens[0],
                    target_tokens[0]
                ], dim=0).unsqueeze(0)

                # Forward pass
                outputs = model(input_ids=input_ids, use_image_prefix=True)
                logits = outputs.logits

                # Calculate target loss
                target_start = head_tokens.shape[1] + candidate_sequences.shape[1] + tail_tokens.shape[1] - 1
                target_end = target_start + target_tokens.shape[1]
                if i == 0:
                    target_logits = logits[0, target_start:target_end, :]
                    target_token_ids, target_decoded = greedy_decode(target_logits, tokenizer)
                    print(f"[DEBUG] Batch i={i} - target section predictions: '{target_decoded}'")
                loss_target[i] = F.cross_entropy(logits[0, target_start:target_end, :], 
                                                target_tokens[0])
            
                # Use soft token loss
                # input_ids = torch.cat([
                #     head_tokens[0], 
                #     candidate_sequences[i], 
                #     tail_tokens[0]
                # ], dim=0).unsqueeze(0)

                # completion_logits, _ = expand_logits_soft(model, input_ids, target_tokens.shape[1], method="gumbel_softmax")
                # loss_target[i] = F.cross_entropy(completion_logits[0], target_tokens[0])


                # print the most likely 5 tokens for the current input
                # test_input_ids = torch.cat([
                #     head_tokens[0],
                #     candidate_sequences[i],
                #     tail_tokens[0]
                # ], dim=0).unsqueeze(0)
                # outputs = model(input_ids=test_input_ids)
                # next_token_logits = outputs.logits[:, -1:, :]  # Shape: [1, 1, vocab_size]
                # top_5_tokens = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).indices
                # top_5_tokens_probs = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).values

                # print("="*80)
                # for j in range(5):
                #     print(f" {j+1}. token: {tokenizer.decode(top_5_tokens[0][j])} with probability {top_5_tokens_probs[0][j].item():.4f}")


    if debug:
        print(f"Sequential evaluation results ({loss_type} loss):")
        
        combined_losses = loss_control + w_tar_2 * loss_target
        best_idx = torch.argmin(combined_losses)
        
        for i in range(min(5, batch_size)):
            sequence_text = tokenizer.decode(candidate_sequences[i], skip_special_tokens=True)
            print(f"  {i+1}. Sequence: '{sequence_text}' | "
                  f"Target Loss: {loss_target[i].item():.4f} | "
                  f"Control Loss: {loss_control[i].item():.4f} | "
                  f"Combined: {combined_losses[i].item():.4f}")
        
        print(f"Best sequence: '{tokenizer.decode(candidate_sequences[best_idx], skip_special_tokens=True)}'")
        print(f"Best sequence target loss: {loss_target[best_idx].item():.4f}")
        print(f"Best sequence control loss: {loss_control[best_idx].item():.4f}")
        print(f"Best sequence combined loss: {combined_losses[best_idx].item():.4f}")

    return loss_target, loss_control 


# def expand_logits(model, prompt_input_ids, max_positions):
#     """Single sequence version for backward compatibility"""
#     with torch.no_grad():
#         current_input = prompt_input_ids
#         completion_logits_list = []
        
#         for _ in range(max_positions):
#             # Get logits for current input
#             outputs = model(input_ids=current_input)
#             next_token_logits = outputs.logits[:, -1:, :]  # Shape: [1, 1, vocab_size]
#             completion_logits_list.append(next_token_logits)

#             # top_5_tokens = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).indices
#             # top_5_tokens_probs = next_token_logits.softmax(dim=-1).squeeze(0).topk(5).values

#             # print("="*80)
#             # for j in range(5):
#             #     print(f" {j+1}. token: {tokenizer.decode(top_5_tokens[0][j])} with probability {top_5_tokens_probs[0][j].item():.4f}")

#             # Get the most likely next token for the next iteration
#             next_token_id = torch.argmax(next_token_logits, dim=-1)  # Shape: [1, 1]
            
#             # Append this token to current input for next iteration
#             current_input = torch.cat([current_input, next_token_id], dim=1)
        
#         # Stack all the logits to get completion_logits
#         completion_logits = torch.cat(completion_logits_list, dim=1)  # Shape: [1, max_positions, vocab_size]

#     return completion_logits


# def expand_logits_batch(model, prompt_input_ids_batch, max_positions):
#     """
#     Batched version of expand_logits for processing multiple sequences simultaneously.
    
#     Args:
#         model: The language model
#         prompt_input_ids_batch: Tensor of shape [batch_size, seq_len] containing input sequences
#         max_positions: Number of positions to expand
        
#     Returns:
#         completion_logits: Tensor of shape [batch_size, max_positions, vocab_size]
#     """
#     with torch.no_grad():
#         batch_size = prompt_input_ids_batch.shape[0]
#         current_input = prompt_input_ids_batch  # Shape: [batch_size, seq_len]
#         completion_logits_list = []
        
#         for _ in range(max_positions):
#             # Get logits for current batch
#             outputs = model(input_ids=current_input)
#             next_token_logits = outputs.logits[:, -1:, :]  # Shape: [batch_size, 1, vocab_size]
#             completion_logits_list.append(next_token_logits)

#             # Get the most likely next token for each sequence in the batch
#             next_token_ids = torch.argmax(next_token_logits, dim=-1)  # Shape: [batch_size, 1]
            
#             # Append these tokens to current input for next iteration
#             current_input = torch.cat([current_input, next_token_ids], dim=1)
        
#         # Stack all the logits to get completion_logits
#         completion_logits = torch.cat(completion_logits_list, dim=1)  # Shape: [batch_size, max_positions, vocab_size]

#     return completion_logits


def logits_to_embedding(logits, embedding_matrix, method="softmax"):
    if method == "softmax":
        probs = F.softmax(logits, dim=-1)
        soft_embedding = torch.matmul(probs, embedding_matrix)
    elif method == "gumbel_softmax":
        probs = F.gumbel_softmax(logits, tau=1.0, hard=False)
        soft_embedding = torch.matmul(probs, embedding_matrix)
    elif method == "top_k":
        k = 5
        top_k_values, top_k_indices = torch.topk(logits, k)
        top_k_probs = F.softmax(top_k_values, dim=-1)
        
        soft_embedding = torch.zeros(embedding_matrix.shape[1], device=logits.device)
        for _, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            soft_embedding += prob * embedding_matrix[idx]
            
    return soft_embedding

def expand_logits_soft(model, prompt_input_ids, max_positions, temperature=1.0, method="softmax"):
    with torch.no_grad():
        embedding_matrix = model.get_input_embeddings().weight  # [vocab_size, embed_dim]
        current_embeds = model.get_input_embeddings()(prompt_input_ids)  # [1, seq_len, embed_dim]
        
        completion_logits_list = []
        soft_embeddings_list = []
        
        for _ in range(max_positions):
            outputs = model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1:, :]  # Shape: [1, 1, vocab_size]
            completion_logits_list.append(next_token_logits)
            scaled_logits = next_token_logits.squeeze(1) / temperature  # [1, vocab_size]

            soft_embedding = logits_to_embedding(scaled_logits, embedding_matrix, method)
            soft_embeddings_list.append(soft_embedding.unsqueeze(1))  # [1, 1, embed_dim]
            
            current_embeds = torch.cat([current_embeds, soft_embedding.unsqueeze(1)], dim=1)
        
        completion_logits = torch.cat(completion_logits_list, dim=1)  # [1, max_positions, vocab_size]
        soft_embeddings = torch.cat(soft_embeddings_list, dim=1)    # [1, max_positions, embed_dim]
    
    return completion_logits, soft_embeddings

def expand_logits_soft_batch(model, prompt_input_ids_batch, max_positions, temperature=1.0, method="gumbel_softmax"):
    with torch.no_grad():
        batch_size = prompt_input_ids_batch.shape[0]
        embedding_matrix = model.get_input_embeddings().weight  # [vocab_size, embed_dim]
        current_embeds = model.get_input_embeddings()(prompt_input_ids_batch)  # [batch_size, seq_len, embed_dim]
        
        completion_logits_list = []
        soft_embeddings_list = []
        
        for _ in range(max_positions):
            outputs = model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            completion_logits_list.append(next_token_logits)
            
            scaled_logits = next_token_logits.squeeze(1) / temperature  # [batch_size, vocab_size]
            
            soft_embeddings = logits_to_embedding(scaled_logits, embedding_matrix, method)
            soft_embeddings_list.append(soft_embeddings.unsqueeze(1))  # [batch_size, 1, embed_dim]
            current_embeds = torch.cat([current_embeds, soft_embeddings.unsqueeze(1)], dim=1)
        
        completion_logits = torch.cat(completion_logits_list, dim=1)  # [batch_size, max_positions, vocab_size]
        soft_embeddings = torch.cat(soft_embeddings_list, dim=1)    # [batch_size, max_positions, embed_dim]
    
    return completion_logits, soft_embeddings

def expand_logits(model, prompt_input_ids, max_positions, **kwargs):
    completion_logits, _ = expand_logits_soft(model, prompt_input_ids, max_positions, **kwargs)
    return completion_logits

def expand_logits_batch(model, prompt_input_ids_batch, max_positions, **kwargs):
    completion_logits, _ = expand_logits_soft_batch(model, prompt_input_ids_batch, max_positions, **kwargs)
    return completion_logits


# COMMENTED OUT - Evaluation function not needed for training
# def log_eval_result(model, tokenizer, attack_prompt, complete_prompt, iter, product_list, target_product, result_table, logger):
#     product_names = [product['Name'] for product in product_list]
# 
#     # Generate result from model using the complete prompt
#     batch_result = model.generate(complete_prompt, model.generation_config, 
#                                   max_new_tokens=500, 
#                                   attention_mask=torch.ones_like(complete_prompt), 
#                                   pad_token_id=tokenizer.eos_token_id)
# 
#     # index batch_result to avoid the input prompt
#     batch_result = batch_result[:, complete_prompt.shape[1]:]
#     generated_texts = tokenizer.batch_decode(batch_result, skip_special_tokens=True)
# 
#     product_ranks = []
#     current_table = wandb.Table(columns=result_table.columns, data=result_table.data)
#     
#     # Evaluate rank for multiple times
#     from experiment.attack import rank_products
#     for i in range(len(complete_prompt)):
#         current_rank = rank_products(generated_texts[i], product_names)[target_product]
#         product_ranks.append(current_rank)     
#         highlighted_attack_prompt = f'<span style="color:red;">{attack_prompt}</span>'
# 
#         log_entry = {
#             "attack_prompt": highlighted_attack_prompt,
#             "complete_prompt": tokenizer.decode(complete_prompt[i], skip_special_tokens=True),
#             "generated_result": generated_texts[i],
#             'product_rank': current_rank
#         }
#         
#         current_table.add_data(iter,
#                         log_entry["attack_prompt"],
#                         log_entry["complete_prompt"],
#                         log_entry["generated_result"],
#                         log_entry["product_rank"])
# 
#     
#     mean_rank = sum(product_ranks) / len(product_ranks)
#         
#     if logger is not None:
#         logger.log({f"eval/target:{target_product}": current_table}, step=iter)
#         logger.log({"eval/product_rank": mean_rank}, step=iter)
#     
#     return current_table, mean_rank