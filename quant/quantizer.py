from typing import Optional, Tuple

import torch

########################## CONSTANTS ##########################

FP4_E2M1_MAX = 6
FP8_E4M3_MAX = 448
NVFP_GROUPSIZE = 16
FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

FP4_GRID =  [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_GRID_TENSOR = torch.tensor(FP4_GRID, dtype=torch.float32)
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10,  9,  8,  0,  1,  2,  3,  4,  5,  6,  7]
FP4_SCALE = 3 / 4

########################## QUANT FUNCTIONS ##########################

def index_rtn_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest index into a sorted 1-D ``grid`` (ties -> higher index)."""
    inds = torch.bucketize(x, grid)
    lo = torch.clamp(inds - 1, min=0, max=grid.shape[-1] - 1)
    hi = torch.clamp(inds, min=0, max=grid.shape[-1] - 1)
    return torch.where((grid[hi] - x) <= (x - grid[lo]), hi, lo)

def quantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    grid = FP4_GRID_TENSOR.to(device=x.device, dtype=x.dtype)
    return grid[index_rtn_grid(x / scales, grid)]

def dequantize_fp4(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    return q.mul(scales)

def quantize_dequantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    xq = dequantize_fp4(quantize_fp4(x, scales, zeros, q_min, q_max), scales, zeros)
    return x + (xq - x).detach()

### Integer Quantization ###
def quantize_int(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int) -> torch.Tensor:
    return (x / scales + zeros).round().clamp(q_min, q_max)

def dequantize_int(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    return q.sub(zeros).mul(scales)

def quantize_dequantize_int(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    xq = dequantize_int(quantize_int(x, scales, zeros, q_min, q_max), scales, zeros)
    return x + (xq - x).detach()


########################## TENSOR HELPERS ##########################

def split_dim(x: torch.Tensor, num_splits: int, dim: int = -1) -> torch.Tensor:
    if dim == -1:
        dim = x.ndim - 1
    new_shape = (
        *x.shape[:dim],
        num_splits,
        x.shape[dim] // num_splits,
        *x.shape[dim + 1 :],
    )
    return x.reshape(new_shape)

def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")

def _resolve_format(fmt: str, bits: int) -> Tuple[callable, int, int]:
    fmt = fmt.lower()
    if fmt in ("nvfp"):
        return quantize_dequantize_fp4, -FP4_E2M1_MAX, FP4_E2M1_MAX
    if fmt.startswith("int"):
        return quantize_dequantize_int, -(2**bits) // 2, (2**bits) // 2 - 1
    raise ValueError(f"Unknown quantization format: {fmt!r}")


class Quantizer:

    def __init__(
        self,
        format: str,
        bits: int = 4,
        symmetric: bool = True,
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_min_clip: Optional[float] = None
    ):
        assert format in ["nvfp", "int"]
        if format == "nvfp": assert bits == 4
        if not symmetric:
            raise NotImplementedError("Asymmetric quantization is not implemented yet.")

        self.format = format
        self.symmetric = symmetric
        self.dim = dim
        self.group_size = group_size
        self.scale_min_clip = scale_min_clip
        self.bits = bits

        self.quant_dequant_fn, self.q_min, self.q_max = _resolve_format(format, self.bits)
        if format == "nvfp":
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32) # needed for NVFP4

    def _reshape_before_quantization(
        self,
        x: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            num_groups = x.shape[dim] // self.group_size
            x = split_dim(x, num_groups, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    def _get_fp4_global_scale(
        self, scales: torch.Tensor, x: torch.Tensor, dynamic: bool
    ) -> torch.Tensor:
        with torch.no_grad():
            current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(x.abs().max().to(torch.float32).view(1))
            if dynamic:
                gs = current_global_scale
                if not gs.isfinite():
                    raise ValueError(f"Global scale is not finite: {gs}\n")
            else:
                if not current_global_scale:
                    raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
                
                self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)
                if not self.global_scale.isfinite():
                    raise ValueError(f"Global scale is not finite: {self.global_scale}\n")
                gs = self.global_scale.to(x.device)

            return ((scales * gs)
                .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                .to(torch.float8_e4m3fn)
                .to(torch.float32)
                .mul(get_reciprocal(gs))
                .to(x.dtype)
            )

    def get_quantization_params(
        self,
        x: torch.Tensor,
        dynamic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        reduce_dim = dim + 1
        x, _, _ = self._reshape_before_quantization(x)

        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)
        absmax = torch.maximum(-x_min, x_max)

        scales = 2 * absmax / (self.q_max - self.q_min)
        zeros = torch.zeros_like(x_min)

        # Reshape back: drop the group axis from the per-group stats.
        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            zeros = zeros.squeeze(dim + 1)

        # NVFP4 phases nest their per-group scales into e4m3 via a global scale.
        if self.format == "nvfp":
            scales = self._get_fp4_global_scale(scales, x, dynamic)

        # Set scales to 1 if zero (avoid division by zero on all-zero groups).
        scales[scales == 0] = 1
        if scales.isnan().any():
            raise ValueError("Scales are not finite.")

        return scales, zeros

    def quantize_dequantize(
        self,
        x: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        xg, s, z = self._reshape_before_quantization(x, scales, zeros)
        xq = self.quant_dequant_fn(xg, s, z, self.q_min, self.q_max)
        return xq.reshape(original_shape)

    def gptq_error(self, w: torch.Tensor, w_q: torch.Tensor, w_d: torch.Tensor) -> torch.Tensor:
        return (w - w_q)


class PrefillDecodeQuantizer(Quantizer):
    """Two individual quantizers: the container is the prefill quantizer and holds an
    independent decode quantizer with its own format and scales."""

    def __init__(
        self,
        format_prefill: str,
        bits_prefill: int = 4,
        format_decode: str = "int",
        bits_decode: int = 3,
        symmetric: bool = True,
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_min_clip: Optional[float] = None,
        err_weight_prefill: float = 0.1,
        err_weight_decode: float = 1.0,
    ):
        super().__init__(
            format_prefill, bits_prefill, symmetric, dim, group_size, scale_min_clip,
            err_weight_prefill, err_weight_decode,
        )
        self.decode_quantizer = Quantizer(format_decode, bits_decode, symmetric, dim, group_size, scale_min_clip)

    def quantize_dequantize_decode(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        xg, s, z = self._reshape_before_quantization(x, scales, zeros)
        xq = self.decode_quantizer.quantize_dequantize(xg, s, z)
        return xq.reshape(original_shape)

    def gptq_error(self, w: torch.Tensor, w_q: torch.Tensor, w_d: torch.Tensor) -> torch.Tensor:
        return (w - w_q) + (w - w_d)

########################## DOWNCAST LUT QUANTIZER ##########################

# MSE-optimal downcast levels: FP4 codes paired by dropping the low index bit (idx >> 1);
# each level is the center of mass of its pair bucket on Gaussian data. See grids.ipynb.
LUT3_DOWNCAST_GRID = [-4.890, -2.550, -1.236, -0.372, 0.372, 1.236, 2.550, 4.890]

class DowncastLUTQuantizer(Quantizer):
    """NVFP4 quantizer whose decode weight is downcast to a 3-bit LUT (fp4 idx >> 1)."""

    def __init__(
        self,
        downcast_grid=LUT3_DOWNCAST_GRID,
        symmetric: bool = True,
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_min_clip: Optional[float] = None,
    ):
        super().__init__(
            "nvfp", 4, symmetric, dim, group_size, scale_min_clip
        )
        self.fp4_grid = torch.tensor(FP4_GRID, dtype=torch.float32)
        self.lut_grid = torch.as_tensor(downcast_grid, dtype=torch.float32)
        # Downcast table: adjacent FP4 codes share a level (16 -> 8), so the zero codes
        # split sign-aware by construction ((-0.5, -0.0) -> level 3, (+0.0, +0.5) -> level 4).
        self.nest = torch.arange(len(FP4_GRID)) >> 1

    def quantize_dequantize_decode(
        self,
        x: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Downcast LUT dequant using the NVFP4 ``scales`` (companion to ``quantize_dequantize``)."""
        original_shape = x.shape
        xg, s, _ = self._reshape_before_quantization(x, scales, zeros)
        fp4_grid = self.fp4_grid.to(x.device)
        lut_grid = self.lut_grid.to(x.device)
        nest = self.nest.to(x.device)
        fp4_idx = index_rtn_grid(xg / s, fp4_grid)
        xq = lut_grid[nest[fp4_idx]] * s
        return xq.reshape(original_shape)


def make_quantizer(scheme: str, wbits: int = 3) -> Quantizer:
    """Build the quantizer for a quantization scheme (shared by the RTN and GPTQ drivers)."""
    if scheme == "nvfp":
        return Quantizer(format="nvfp", bits=4, symmetric=True, group_size=NVFP_GROUPSIZE)
    if scheme == "int":
        return Quantizer(format="int", bits=wbits, symmetric=True, group_size=NVFP_GROUPSIZE)
    if scheme == "independent":
        return PrefillDecodeQuantizer("nvfp", 4, "int", wbits, group_size=NVFP_GROUPSIZE)
    if scheme == "downcast":
        return DowncastLUTQuantizer(group_size=NVFP_GROUPSIZE)
    raise ValueError(f"Unknown quantization scheme: {scheme!r}")
