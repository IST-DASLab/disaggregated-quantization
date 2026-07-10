"""
Metrics for evaluating quantized models: perplexity, KL divergence, and acceptance probability.
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm


@dataclass
class GeneratedSequence:
    """A single generated sequence with draft/verify structure."""
    prompt_ids: torch.Tensor
    draft_chunks: List[torch.Tensor]
    fp16_accepted_counts: List[int]
    fp16_corrections: List[torch.Tensor]
    total_drafted: int
    total_fp16_accepted: int


@torch.no_grad()
def compute_perplexity(
    model,
    data: List[torch.Tensor],
    batch_size: int = 1,
    masks: Optional[List[torch.Tensor]] = None,
) -> Tuple[float, float, List[float]]:
    """
    Compute perplexity over a dataset with per-sample tracking.
    
    Args:
        model: Model to evaluate
        data: List of input tensors
        batch_size: Batch size for inference
        masks: Optional list of attention masks (1=valid, 0=pad)
    
    Returns:
        (mean_perplexity, variance_perplexity, per_sample_ppl) tuple
    """
    num_samples = len(data)
    device = next(model.parameters()).device
    use_masks = masks is not None
    
    per_sample_nll = []
    
    for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        )
        loss_per_seq = loss.reshape(shift_labels.shape)
        
        if use_masks:
            batch_masks = torch.cat(masks[i:j]).to(device)
            shift_masks = batch_masks[:, 1:].float()
            masked_loss = loss_per_seq * shift_masks
            valid_counts = shift_masks.sum(dim=1).clamp(min=1)
            nll_per_seq = masked_loss.sum(dim=1) / valid_counts
        else:
            nll_per_seq = loss_per_seq.mean(dim=1)
        
        per_sample_nll.extend(nll_per_seq.cpu().tolist())
    
    mean_nll = np.mean(per_sample_nll)
    mean_ppl = np.exp(mean_nll)
    per_sample_ppl = [np.exp(nll) for nll in per_sample_nll]
    var_ppl = np.var(per_sample_ppl)
    
    return mean_ppl, var_ppl, per_sample_ppl


@torch.no_grad()
def compute_baseline_topk(
    model,
    input_ids: torch.Tensor,
    batch_size: int = 4,
    top_k: int = 10,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k baseline logits for KL divergence comparison.
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs of shape [num_samples, seq_len]
        batch_size: Batch size for inference
        top_k: Number of top tokens to store
        masks: Optional attention masks of shape [num_samples, seq_len]
    
    Returns:
        (topk_vals, topk_idx) tuple, each of shape [num_samples, seq_len-1, top_k]
    """
    model.eval()
    device = next(model.parameters()).device
    all_topk_vals = []
    all_topk_idx = []
    
    for i in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[i:i + batch_size].to(device)
        logits = model(batch).logits[:, :-1, :].float()
        topk_vals, topk_idx = logits.topk(top_k, dim=-1)
        all_topk_vals.append(topk_vals.cpu())
        all_topk_idx.append(topk_idx.cpu())
        del logits
        torch.cuda.empty_cache()
    
    return torch.cat(all_topk_vals, dim=0), torch.cat(all_topk_idx, dim=0)


@torch.no_grad()
def compute_all_metrics(
    model,
    calibration_data: torch.Tensor,
    baseline_topk_vals: torch.Tensor,
    baseline_topk_idx: torch.Tensor,
    batch_size: int = 4,
    top_k: int = 10,
    temperature: float = 1.0,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """
    Compute NLL, KL divergence, and EAR in a single forward pass.
    
    Args:
        model: Model to evaluate
        calibration_data: Input token IDs
        baseline_topk_vals: Pre-computed baseline top-k logit values
        baseline_topk_idx: Pre-computed baseline top-k logit indices
        batch_size: Batch size for inference
        top_k: Number of top tokens
        temperature: Temperature for softmax
        masks: Optional attention masks [num_samples, seq_len], 1=valid, 0=pad
    
    Returns:
        (nll, kl_divergence, ear) tuple
    """
    model.eval()
    device = next(model.parameters()).device
    use_masks = masks is not None
    
    total_nll = 0.0
    total_kl = 0.0
    total_ear = 0.0
    total_tokens = 0
    
    batch_idx = 0
    for i in range(0, calibration_data.shape[0], batch_size):
        batch = calibration_data[i:i + batch_size].to(device)
        actual_bs = batch.shape[0]
        
        if use_masks:
            batch_masks = masks[i:i + batch_size].to(device)
            shift_masks = batch_masks[:, 1:].float()
        
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        # NLL
        nll_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(actual_bs, -1)
        
        if use_masks:
            total_nll += (nll_loss * shift_masks).sum().item()
        else:
            total_nll += nll_loss.sum().item()
        
        # KL and EAR
        draft_logits = shift_logits.float() / temperature
        vtl = baseline_topk_vals[batch_idx:batch_idx + actual_bs].to(device) / temperature
        vti = baseline_topk_idx[batch_idx:batch_idx + actual_bs].to(device)
        
        for pos_start in range(0, draft_logits.shape[1], 256):
            pos_end = min(pos_start + 256, draft_logits.shape[1])
            dc = draft_logits[:, pos_start:pos_end, :]
            vtl_chunk = vtl[:, pos_start:pos_end, :]
            vti_chunk = vti[:, pos_start:pos_end, :]
            
            dtl = dc.gather(dim=-1, index=vti_chunk)
            
            vtp = F.softmax(vtl_chunk, dim=-1)
            dtp = F.softmax(dtl, dim=-1)
            
            eps = 1e-10
            kl = (vtp * (torch.log(vtp + eps) - torch.log(dtp + eps))).sum(dim=-1)
            ear = (vtp * torch.clamp(dtp / (vtp + eps), max=1.0)).sum(dim=-1)
            
            if use_masks:
                chunk_masks = shift_masks[:, pos_start:pos_end]
                total_kl += (kl * chunk_masks).sum().item()
                total_ear += (ear * chunk_masks).sum().item()
                total_tokens += chunk_masks.sum().item()
            else:
                total_kl += kl.sum().item()
                total_ear += ear.sum().item()
                total_tokens += kl.numel()
        
        batch_idx += actual_bs
        torch.cuda.empty_cache()
    
    n = total_tokens if total_tokens > 0 else 1
    return total_nll / n, total_kl / n, total_ear / n


@torch.no_grad()
def compute_kl_and_ear(
    model,
    baseline_topk_vals: torch.Tensor,
    baseline_topk_idx: torch.Tensor,
    eval_data: torch.Tensor,
    batch_size: int = 4,
    return_per_sample: bool = False,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, Dict]:
    """
    Compute top-k KL divergence and Expected Acceptance Rate (EAR).
    
    Args:
        model: Model to evaluate
        baseline_topk_vals: Pre-computed baseline top-k logit values
        baseline_topk_idx: Pre-computed baseline top-k logit indices
        eval_data: Input token IDs [num_samples, seq_len]
        batch_size: Batch size for inference
        return_per_sample: Whether to return per-sample KL values
        masks: Optional attention masks [num_samples, seq_len], 1=valid, 0=pad
    
    Returns:
        (mean_kl, mean_ear, result_dict) tuple
    """
    model.eval()
    device = next(model.parameters()).device
    use_masks = masks is not None
    
    total_kl, total_ear, total_tokens = 0.0, 0.0, 0
    
    all_kl_values = []
    per_sample_kl = [] if return_per_sample else None

    batch_idx = 0
    for i in range(0, eval_data.shape[0], batch_size):
        batch = eval_data[i:i + batch_size].to(device)
        actual_bs = batch.shape[0]
        
        if use_masks:
            batch_masks = masks[i:i + batch_size].to(device)
            shift_masks = batch_masks[:, 1:].float()

        quant_logits = model(batch).logits[:, :-1, :].float()
        
        ftl = baseline_topk_vals[batch_idx:batch_idx + actual_bs].to(device)
        fti = baseline_topk_idx[batch_idx:batch_idx + actual_bs].to(device)

        sample_kl_sums = torch.zeros(actual_bs, device=device)
        sample_token_counts = torch.zeros(actual_bs, device=device)

        for pos_start in range(0, quant_logits.shape[1], 256):
            pos_end = min(pos_start + 256, quant_logits.shape[1])
            ql = quant_logits[:, pos_start:pos_end, :]
            ftl_chunk = ftl[:, pos_start:pos_end, :]
            fti_chunk = fti[:, pos_start:pos_end, :]

            qtl = ql.gather(dim=-1, index=fti_chunk)
            
            fp = F.softmax(ftl_chunk, dim=-1)
            qp = F.softmax(qtl, dim=-1)
            
            eps = 1e-10
            
            kl = (fp * (torch.log(fp + eps) - torch.log(qp + eps))).sum(dim=-1)
            ear = (fp * torch.clamp(qp / (fp + eps), max=1.0)).sum(dim=-1)
            
            if use_masks:
                chunk_masks = shift_masks[:, pos_start:pos_end]
                masked_kl = kl * chunk_masks
                masked_ear = ear * chunk_masks
                
                total_kl += masked_kl.sum().item()
                total_ear += masked_ear.sum().item()
                total_tokens += chunk_masks.sum().item()
                
                all_kl_values.extend(kl[chunk_masks.bool()].cpu().tolist())
                
                if return_per_sample:
                    sample_kl_sums += masked_kl.sum(dim=1)
                    sample_token_counts += chunk_masks.sum(dim=1)
            else:
                total_kl += kl.sum().item()
                total_ear += ear.sum().item()
                total_tokens += kl.numel()
                
                all_kl_values.extend(kl.flatten().cpu().tolist())
                
                if return_per_sample:
                    sample_kl_sums += kl.sum(dim=1)
                    sample_token_counts += kl.shape[1]

        if return_per_sample:
            valid_counts = sample_token_counts.clamp(min=1)
            per_sample_kl.extend((sample_kl_sums / valid_counts).cpu().tolist())

        batch_idx += actual_bs
        torch.cuda.empty_cache()

    mean_kl = total_kl / total_tokens if total_tokens > 0 else float('inf')
    mean_ear = total_ear / total_tokens if total_tokens > 0 else 0.0
    kl_p95 = float(np.percentile(all_kl_values, 95)) if all_kl_values else 0.0
    kl_var = float(np.var(all_kl_values)) if all_kl_values else 0.0
    
    result = {
        'mean_kl': mean_kl,
        'mean_ear': mean_ear,
        'kl_p95': kl_p95,
        'kl_variance': kl_var,
        'total_tokens': total_tokens,
    }
    
    if return_per_sample:
        result['per_sample_kl'] = per_sample_kl
    
    return mean_kl, mean_ear, result



def _get_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Get next token from logits (greedy or sampled)."""
    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return logits.argmax(dim=-1, keepdim=True)


def _count_accepted_tokens(
    verify_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_len: int,
    temperature: float,
) -> int:
    """
    Count how many consecutive draft tokens are accepted by the verifier.
    """
    accepted = 0
    for i in range(draft_len):
        v_token = _get_next_token(verify_logits[:, i, :], temperature).squeeze(-1)
        
        if v_token.item() == draft_tokens[0, i].item():
            accepted += 1
        else:
            break
    
    return accepted


@torch.no_grad()
def generate_anchor_sequences(
    fp16_model,
    drafter,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    draft_len: int = 5,
    temperature: float = 0.0,
) -> List[GeneratedSequence]:
    """
    Generate anchor sequences using FP16 model.
    
    These sequences will be used to evaluate all models (including FP16 itself)
    to ensure identical comparison conditions.
    
    Returns:
        List of GeneratedSequence objects, one per prompt
    """
    fp16_model.eval()
    drafter.eval()
    
    sequences = []
    
    for prompt in tqdm(prompts, desc="Generating anchor sequences", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        prompt_ids = inputs.input_ids.clone()
        input_ids = inputs.input_ids.to(fp16_model.device)
        
        draft_chunks = []
        fp16_accepted_counts = []
        fp16_corrections = []
        total_drafted = 0
        total_fp16_accepted = 0
        generated = 0
        
        while generated < max_new_tokens:
            # Draft phase
            drafter_ids = input_ids.to(drafter.device)
            
            for _ in range(draft_len):
                drafter_out = drafter(drafter_ids)
                logits = drafter_out.logits[:, -1, :]
                next_token = _get_next_token(logits, temperature)
                drafter_ids = torch.cat([drafter_ids, next_token], dim=-1)
            
            draft_tokens = drafter_ids[:, input_ids.shape[1]:].to(fp16_model.device)
            draft_chunks.append(draft_tokens.cpu().clone())
            total_drafted += draft_len
            
            # Verify with FP16
            verify_input = torch.cat([input_ids, draft_tokens], dim=1)
            fp16_out = fp16_model(verify_input)
            fp16_verify_logits = fp16_out.logits[:, input_ids.shape[1]-1:-1, :]
            
            fp16_accepted = _count_accepted_tokens(
                fp16_verify_logits, draft_tokens, draft_len, temperature
            )
            fp16_accepted_counts.append(fp16_accepted)
            total_fp16_accepted += fp16_accepted
            
            # Update sequence based on FP16's decisions
            if fp16_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens[:, :fp16_accepted]], dim=1)
                generated += fp16_accepted
            
            # Store and apply FP16's correction if needed
            if fp16_accepted < draft_len:
                correction = fp16_verify_logits[:, fp16_accepted, :].argmax(dim=-1, keepdim=True)
                fp16_corrections.append(correction.cpu().clone())
                input_ids = torch.cat([input_ids, correction], dim=1)
                generated += 1
            else:
                fp16_corrections.append(None)
            
            if generated >= max_new_tokens:
                break
        
        sequences.append(GeneratedSequence(
            prompt_ids=prompt_ids,
            draft_chunks=draft_chunks,
            fp16_accepted_counts=fp16_accepted_counts,
            fp16_corrections=fp16_corrections,
            total_drafted=total_drafted,
            total_fp16_accepted=total_fp16_accepted,
        ))
    
    return sequences


@torch.no_grad()
def evaluate_model_on_sequences(
    model,
    sequences: List[GeneratedSequence],
    draft_len: int = 5,
    temperature: float = 0.0,
) -> Dict:
    """
    Evaluate a model on pre-generated anchor sequences.
    
    The model sees the exact same sequences as all other models,
    enabling clean paired comparisons.
    
    Args:
        model: Model to evaluate (can be FP16 or quantized)
        sequences: Pre-generated anchor sequences from FP16
        draft_len: Draft length used during sequence generation
        temperature: Temperature used during sequence generation
    
    Returns:
        Dict with acceptance metrics, ETL, and per-prompt values
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_model_accepted = 0
    total_fp16_accepted = 0
    total_drafted = 0
    total_token_loss = 0
    
    per_prompt_acceptance = []
    per_prompt_etl = []
    
    for seq in tqdm(sequences, desc="Evaluating on sequences", leave=False):
        input_ids = seq.prompt_ids.to(device)
        
        prompt_model_accepted = 0
        prompt_fp16_accepted = 0
        prompt_drafted = 0
        prompt_token_loss = 0
        
        for chunk_idx, draft_tokens in enumerate(seq.draft_chunks):
            draft_tokens = draft_tokens.to(device)
            fp16_accepted = seq.fp16_accepted_counts[chunk_idx]
            correction = seq.fp16_corrections[chunk_idx]
            
            prompt_drafted += draft_len
            prompt_fp16_accepted += fp16_accepted
            
            # Verify with this model
            verify_input = torch.cat([input_ids, draft_tokens], dim=1)
            model_out = model(verify_input)
            model_verify_logits = model_out.logits[:, input_ids.shape[1]-1:-1, :]
            
            model_accepted = _count_accepted_tokens(
                model_verify_logits, draft_tokens, draft_len, temperature
            )
            prompt_model_accepted += model_accepted
            
            # ETL: tokens FP16 accepted but this model rejected
            token_loss = max(0, fp16_accepted - model_accepted)
            prompt_token_loss += token_loss
            
            # Follow FP16's path (same for all models)
            if fp16_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens[:, :fp16_accepted]], dim=1)
            
            if correction is not None:
                input_ids = torch.cat([input_ids, correction.to(device)], dim=1)
        
        # Accumulate totals
        total_model_accepted += prompt_model_accepted
        total_fp16_accepted += prompt_fp16_accepted
        total_drafted += prompt_drafted
        total_token_loss += prompt_token_loss
        
        # Per-prompt metrics
        if prompt_drafted > 0:
            per_prompt_acceptance.append(prompt_model_accepted / prompt_drafted)
        else:
            per_prompt_acceptance.append(0.0)
        
        if prompt_fp16_accepted > 0:
            per_prompt_etl.append(prompt_token_loss / prompt_fp16_accepted)
        else:
            per_prompt_etl.append(0.0)
    
    # Compute aggregate metrics
    acceptance_rate = total_model_accepted / total_drafted if total_drafted > 0 else 0.0
    fp16_acceptance_rate = total_fp16_accepted / total_drafted if total_drafted > 0 else 0.0
    etl = total_token_loss / total_fp16_accepted if total_fp16_accepted > 0 else 0.0
    
    # Compute statistics
    def compute_stats(values):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        variance = float(np.var(values))
        std = float(np.std(values))
        se = std / np.sqrt(len(values))
        ci95 = 1.96 * se
        return variance, std, se, ci95
    
    acc_var, acc_std, acc_se, acc_ci95 = compute_stats(per_prompt_acceptance)
    etl_var, etl_std, etl_se, etl_ci95 = compute_stats(per_prompt_etl)
    
    return {
        # Model acceptance
        'acceptance_rate': acceptance_rate,
        'acceptance_rate_variance': acc_var,
        'acceptance_rate_std': acc_std,
        'acceptance_rate_se': acc_se,
        'acceptance_rate_ci95': acc_ci95,
        'per_prompt_acceptance': per_prompt_acceptance,
        
        # FP16 reference (same for all models)
        'fp16_acceptance_rate': fp16_acceptance_rate,
        
        # ETL
        'etl': etl,
        'etl_variance': etl_var,
        'etl_std': etl_std,
        'etl_se': etl_se,
        'etl_ci95': etl_ci95,
        'per_prompt_etl': per_prompt_etl,
        
        # Totals
        'total_model_accepted': total_model_accepted,
        'total_fp16_accepted': total_fp16_accepted,
        'total_drafted': total_drafted,
        'total_token_loss': total_token_loss,
    }


@torch.no_grad()
def compute_conditional_perplexity(
    model,
    data: List[torch.Tensor],
    loss_masks: List[torch.Tensor],
    batch_size: int = 1,
) -> float:
    """
    Compute conditional perplexity using loss masks.
    """
    num_samples = len(data)
    device = next(model.parameters()).device

    nll_running = 0.0
    answer_tokens_processed = 0

    for i in trange(0, num_samples, batch_size, desc="Computing conditional perplexity", leave=False):
        j = min(i + batch_size, num_samples)

        inputs = torch.cat(data[i:j]).to(device)
        masks = torch.cat(loss_masks[i:j]).to(device)

        outputs = model(inputs)
        lm_logits = outputs.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:].contiguous()
        shift_mask = masks[:, 1:].contiguous()

        loss_unreduced = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        )

        loss_unreduced = loss_unreduced.reshape_as(shift_labels)
        masked_losses = loss_unreduced * shift_mask

        batch_answer_tokens = shift_mask.sum().item()

        if batch_answer_tokens > 0:
            batch_loss = masked_losses.sum() / batch_answer_tokens

            a = batch_answer_tokens / (answer_tokens_processed + batch_answer_tokens)
            b = answer_tokens_processed / (answer_tokens_processed + batch_answer_tokens)
            nll_running = a * batch_loss + b * nll_running

            answer_tokens_processed += batch_answer_tokens

    if answer_tokens_processed == 0:
        raise ValueError("No answer tokens found in the data")

    return torch.exp(torch.tensor(nll_running)).item()


