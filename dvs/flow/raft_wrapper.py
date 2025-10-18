import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)


LOGGER = logging.getLogger(__name__)


class RAFTFlowEstimator:
    """Lightweight wrapper around torchvision's RAFT implementation.

    The estimator exposes a simple API that accepts a sequence of frames and
    returns forward and backward optical flow tensors shaped like the legacy
    FlowNet2 outputs used by the stabilization network.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_variant: str = "small",
        use_pretrained: bool = True,
        progress: bool = True,
    ) -> None:
        self.model_variant = model_variant.lower()
        self.use_pretrained = use_pretrained
        self.progress = progress
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[torch.nn.Module] = None
        self._weights_name: Optional[str] = None
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if self.model_variant == "large":
            build_fn = raft_large
            weights_enum = Raft_Large_Weights.DEFAULT if self.use_pretrained else None
        else:
            if self.model_variant != "small":
                LOGGER.warning(
                    "Unknown RAFT variant '%s'. Falling back to 'small'.",
                    self.model_variant,
                )
            build_fn = raft_small
            weights_enum = Raft_Small_Weights.DEFAULT if self.use_pretrained else None

        self._weights_name = weights_enum.name if weights_enum is not None else None
        self._model = build_fn(weights=weights_enum, progress=self.progress)
        self._model.to(self.device)
        self._model.eval()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Lazily rebuild the model when the estimator is used again.
        self._ensure_model()

    @property
    def model(self) -> torch.nn.Module:
        self._ensure_model()
        assert self._model is not None
        return self._model

    def _prepare_tensor(self, frame: np.ndarray) -> torch.Tensor:
        if frame.dtype != np.float32:
            tensor = torch.from_numpy(frame.astype(np.float32))
        else:
            tensor = torch.from_numpy(frame)
        tensor = tensor.permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).to(self.device)

    def _run_model(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            output = self.model(tensor1, tensor2)[-1]
        flow = output[0].permute(1, 2, 0).detach().cpu().numpy()
        return flow.astype(np.float32)

    def compute_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        tensor1 = self._prepare_tensor(frame1)
        tensor2 = self._prepare_tensor(frame2)
        return self._run_model(tensor1, tensor2)

    def compute_sequence(
        self, frames: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(frames) < 2:
            return (
                np.zeros((0, 0, 0, 2), dtype=np.float32),
                np.zeros((0, 0, 0, 2), dtype=np.float32),
            )

        tensors: List[torch.Tensor] = [self._prepare_tensor(frame) for frame in frames]
        flows_forward = []
        flows_backward = []
        for i in range(len(tensors) - 1):
            flow_fw = self._run_model(tensors[i], tensors[i + 1])
            flow_bw = self._run_model(tensors[i + 1], tensors[i])
            flows_forward.append(flow_fw)
            flows_backward.append(flow_bw)

        return np.stack(flows_forward, axis=0), np.stack(flows_backward, axis=0)


__all__ = ["RAFTFlowEstimator"]
