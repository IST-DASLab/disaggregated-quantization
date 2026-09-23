# Disaggregated Quantization: Better Answers and Faster Prefill for Low-Bit LLMs

A local LLM has two distinct inference phases. **Prefill** processes the input and builds the KV-caches that condition the answer. **Decode** uses those KV-caches to generate new tokens, one at a time. 
Decode uses each weight exactly once, making the movement of weights from memory the bottleneck.
Prefill, on the contrary, reuses almost evey weight across the entire input sequence, making weight-loading asymptotically insignifican relative to compute for longer context.

**Disaggregated quantization (DQ)** exploits this dichotomy to improve local LLM deployment in three steps:

 1. Use an extremely **low-bitwidth weight-only** quantization for the **decode-focused checkpoint** to reduce device occupation and memory traffic on decode.
 2. Perform prefill with a **separate set of hardware-native (NVFP4) weights** for both accelerated computation and accuracy gains.
 3. **Offload the prefill-specific weights** to SSD and **overlap their loading with computations** for long-context sequence processing to fully negate the extra weights storage while still accelerating prefill at long sequences.

The practical result is that an **NVFP4 prefill conjugate** can be trained for any existing low-bitwidth weight-only checkpoint to both significantly boost its accuracy and accelerate long-context prefill at no extra memory cost.

<video autoplay loop muted playsinline controls width="100%" poster="https://huggingface.co/datasets/apanferovnvidia/disaggregated-quantization-blog-assets/resolve/main/odp_mechanism_poster.png">
  <source src="https://huggingface.co/datasets/apanferovnvidia/disaggregated-quantization-blog-assets/resolve/main/odp_mechanism.mp4" type="video/mp4">
</video>

*Offloaded Disaggregated Prefill (ODP) loads and computes with prefill blocks in an overlapping pipeline, then hands generation to the resident weight-only decoder.*

## Quick Links

- [Technical report on arXiv](TODO)
- [Training and evalutaion codebase](TODO)
- [Llama.cpp fork with ODP support](TODO)
- [Trained NVFP4 prefill conjugates for Qwen3.8-27B](TODO)

## Two Phases, Different Quantization Choices

For decode, a compact weight format can be useful even when the accelerator cannot directly operate in that format. A CUDA kernel dequantizes the weights as it loads them, reducing memory traffic while keeping activations and performing operations in higher precision. This supports a broad ecosystem of weight-only formats, including encodings that represent groups of weights together, such as vector or trellis encodings.

Prefill is different. A long prompt gives the accelerator enough parallel work to benefit from quantized general matrix multiplication (GEMM) operations. The benefits of such operations, however, stems from hardware support for quantized GEMM instructions, and requies activations to be quantized as well. NVIDIA Blackwell chips provide such support for NVFP4-quantized tensors.

The strength of Disaggregated Quantization (DQ) is in handling these two regimes simultaneously yet differently but utilizing a native NVFP4 checkpoint for prefill and a more complex but lower-bitwidth weight-only checkpoint for decode.

## Performance Highlights: Improving an Existing Decoder

One of the most straight-forward applications of DQ is training an **NVFP4 prefill conjugate** to existing weight-only checkpoint. This conjugate checkpoint is tailored to each pre-quantized weight-only checkpoint to replace its computation specifically on prefill.

For **Qwen3.8-27B**, we train NVFP4 prefill conjugates for released Unsloth GGUF checkpoints while leaving their decode weights and computations untouched.

| Decode format | MMLU-Pro: weight-only → conjugate | MMMU-Pro: weight-only → conjugate |
|---|---:|---:|
| IQ1_S | 29.04% → 61.54% | 24.39% → 59.65% |
| IQ1_M | 52.88% → 72.59% | 44.86% → 62.49% |
| IQ2_XXS | 70.50% → 77.93% | 59.60% → 65.90% |

For IQ1_S, NVFP4 prefill conjugate boosts accuracy by **32.5% points on text reasoning and 35.3% on visual reasoning**. IQ2_XXS, a higher-bitwidth vector-quantized format, also benefits: gains are 7.4% and 6.3%, respectively.

<video autoplay loop muted playsinline controls width="100%" poster="https://huggingface.co/datasets/apanferovnvidia/disaggregated-quantization-blog-assets/resolve/main/dq_results_poster.png">
  <source src="https://huggingface.co/datasets/apanferovnvidia/disaggregated-quantization-blog-assets/resolve/main/dq_results.mp4" type="video/mp4">
</video>

*The decoder stays unchanged. The trained prefill conjugate improves the KV-caches it receives. The size axis describes encoded GGUF backbone weights, not full llama.cpp's allocations.*

## Training a Prefill Conjugate

The interface between prefill and decode is the model's KV-caches. Prefill computes them, and decode attends to them while generating the answer. A prefill conjugate learns to produce representations that work well with its decoder, retaining the original attention architecture.

We train this using **quantization-aware distillation with disaggregation (QADD)**. A frozen BF16 teacher provides target output distributions. In the student, prompt positions use the prefill pathway and response positions use the decode pathway. The same token mask that identifies assistant responses in instruction-tuning data also selects which pathway each position uses.

The objective only scores the response, but its gradients reach context through the KV-caches. Both pathways are therefore optimized toward the same answer in one forward-backward pass.

![QADD routes prompt and response positions through phase-specific quantized pathways.](https://huggingface.co/datasets/apanferovnvidia/disaggregated-quantization-blog-assets/resolve/main/qadd_training_blog.png)

For an existing GGUF checkpoint, we freeze the decoder and train only its prefill conjugate. Gradients still propagate through the decoder's computations, using its dequantized weights, but neither its weights nor its quantization procedure needs to be updated. This matters for complex encodings and for checkpoints produced using training data or optimization methods that are not available to the user.

## Serving the Extra Checkpoint Locally

A separate prefill model is expensive to keep around. But its weights are only needed while processing the prompt. Once a transformer block has produced its outputs, those weights can be discarded: the KV-caches representations remain available to decode and the activations can pass further up the network.

ODP uses two reusable block buffers to overlap loading and computation. While one block processes the prompt, the next is loaded from SSD. The buffers borrow device memory from a portion of the decode checkpoint, which is unused during prefill and restored before generation. During decode, only the compressed decode model is resident and the prefill buffers are no longer needed. This nullifies the impact of prefill weights on device memory, limiting their persistent impact to disk space.

Longer prompts provide more computation over which to hide loading. We implemented this pathway in a custom fork of **llama.cpp**, retaining its native weight-only decode processing. On DGX Spark with Qwen3.8-27B and Unsloth IQ1_S:

- At **8K context**, time to first token falls from **12.27 to 6.90 seconds**, a **1.78× speedup**.
- Across measured **4K–32K contexts**, speedups range from **1.38× to 1.78×**.
- At **1K context**, SSD loading dominates: NVFP4 ODP takes **3.22 seconds**, versus **1.17 seconds** for non-offloaded BF16 computations.

## Broader Validation

Broader and larger-scale experiments are compiled into the technical report available on [arXiv](TODO).

The broader experiments cover Qwen 3 and Gemma 3 with 2–4-bit decode formats. We evaluate both decode-heavy reasoning tasks, where the model generates substantial output, and prefill-heavy long-context tasks, where it processes a long input to produce a short answer.

Removing decode activation quantization primarily helps decode-heavy workloads. Giving prefill its own weights improves low-bit accuracy on both workload types. At 2-bit decode, full disaggregation exceeds trained weight-only baselines by 7.1 and 4.5 points in average decode-heavy accuracy for Qwen 3 and Gemma 3, respectively. On prefill-heavy tasks, those gains grow to 12.6 and 8.9 points.

The simpler shared-weight intervention also carries over to larger models. Without retraining, format disaggregation improves accuracy in 11 of 13 text- and visual-reasoning comparisons across models up to 2.8 trillion parameters, with six statistically significant gains and no significant degradations under per-comparison paired tests. This large-scale validation tests format disaggregation, not full conjugate training or ODP.

## A New Way to Serve Quantized Models

Disaggregated Quantization decouples computational pathways, model weights and their placement on each phase of LLM serving to yield simultaneously more accurate and faster model for local deployment.

The clearest demonstrated use case is local, low-batch inference with aggressively compressed weights and sufficiently long prompts. Highly batched serving, tool usage and multi-turn agentic workloads remain open research directions.
