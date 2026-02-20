"""
Projected Gradient Descent (PGD) Attack for Face Privacy Protection

This is the core attack implementation that combines:
  1. PGD optimization (Madry et al. 2018)
  2. EoT for compression robustness (Athalye et al. 2018)
  3. Face embedding distance maximization (inspired by PhotoGuard/Anti-DreamBooth)

Attack objective (untargeted):
    max_δ  𝔼_{t~T} [ 1 - cos(F(t(x+δ)), F(x)) ]
    s.t.   ‖δ‖∞ ≤ ε,  x+δ ∈ [0, 1]

Attack objective (targeted):
    min_δ  𝔼_{t~T} [ cos(F(t(x+δ)), e_target) ]
    s.t.   ‖δ‖∞ ≤ ε,  x+δ ∈ [0, 1]

The ASPL-style alternation from Anti-DreamBooth is also available:
iterate between updating the surrogate and the perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
from tqdm import tqdm

from .eot import EoTWrapper
from .face_model import FaceEmbedder


class PGDAttack:
    """
    PGD-based adversarial perturbation for face privacy.

    Usage:
        attack = PGDAttack(face_model, eot_wrapper, config)
        x_protected = attack.run(x_aligned)
    """

    def __init__(
        self,
        face_model: FaceEmbedder,
        eot_wrapper: EoTWrapper,
        epsilon: float = 8 / 255,
        step_size: float = 2 / 255,
        num_steps: int = 50,
        random_start: bool = True,
        targeted: bool = False,
        verbose: bool = True,
    ):
        self.face_model = face_model
        self.eot = eot_wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.targeted = targeted
        self.verbose = verbose

    @torch.no_grad()
    def _get_clean_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding of clean image (no transforms, no grad)."""
        return self.face_model(x)

    def run(
        self,
        x: torch.Tensor,
        target_embedding: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, float], None]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run PGD attack on aligned face image(s).

        Args:
            x: (B, 3, 112, 112) aligned face tensor in [0, 1]
            target_embedding: (B, 512) for targeted attack (push toward this identity)
            callback: optional fn(step, x_adv, loss) called each iteration

        Returns:
            x_adv: (B, 3, 112, 112) protected image in [0, 1]
            info: dict with loss history and cosine similarity metrics
        """
        device = x.device
        x = x.detach().clone()

        # Get clean embedding (attack anchor)
        clean_emb = self._get_clean_embedding(x)

        # Initialize perturbation
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            delta = delta.clamp(0 - x, 1 - x)  # Keep x + delta in [0, 1]
        else:
            delta = torch.zeros_like(x)

        delta.requires_grad_(True)

        loss_history = []
        cos_history = []

        iterator = range(self.num_steps)
        if self.verbose:
            iterator = tqdm(iterator, desc="PGD Attack", leave=False)

        for step in iterator:
            # Forward: compute EoT loss
            x_adv = (x + delta).clamp(0.0, 1.0)

            if self.targeted and target_embedding is not None:
                # Targeted: minimize distance to target
                loss = self.eot(x_adv, target_embedding)
            else:
                # Untargeted: minimize cosine sim = maximize distance
                loss = self.eot(x_adv, clean_emb)

            # Backward
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach()

                if self.targeted:
                    # Targeted: descend toward target (minimize sim to target)
                    delta_update = -self.step_size * grad.sign()
                else:
                    # Untargeted: descend to minimize cosine sim (push apart)
                    delta_update = -self.step_size * grad.sign()

                # Update delta
                delta.data = delta.data + delta_update

                # Project onto ε-ball (L∞)
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)

                # Project onto valid image range
                delta.data = (x + delta.data).clamp(0.0, 1.0) - x

                # Logging
                loss_val = loss.item()
                loss_history.append(loss_val)

                # Compute current cosine similarity for monitoring
                with torch.no_grad():
                    adv_emb = self.face_model((x + delta.data).clamp(0.0, 1.0))
                    cos_sim = F.cosine_similarity(clean_emb, adv_emb, dim=1).mean().item()
                    cos_history.append(cos_sim)

                if self.verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(
                        loss=f"{loss_val:.4f}",
                        cos_sim=f"{cos_sim:.4f}",
                        delta_norm=f"{delta.data.abs().max():.4f}",
                    )

                if callback:
                    callback(step, (x + delta.data).clamp(0.0, 1.0), loss_val)

            # Reset grad for next step
            delta.grad.zero_()

        # Final result
        x_adv = (x + delta.data).clamp(0.0, 1.0)

        # Final evaluation with more EoT samples
        with torch.no_grad():
            final_emb = self.face_model(x_adv)
            final_cos = F.cosine_similarity(clean_emb, final_emb, dim=1).mean().item()

            # Also measure with transforms
            robust_emb = self.eot.get_transformed_embedding(x_adv, num_avg=30)
            robust_cos = F.cosine_similarity(clean_emb, robust_emb, dim=1).mean().item()

        info = {
            "loss_history": loss_history,
            "cosine_history": cos_history,
            "final_cosine_sim": final_cos,
            "robust_cosine_sim": robust_cos,
            "delta_linf": delta.data.abs().max().item(),
            "delta_l2": delta.data.norm(p=2).item(),
            "num_steps": self.num_steps,
        }

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"PGD Attack Complete")
            print(f"  Clean→Adv cosine sim: {final_cos:.4f}")
            print(f"  Robust cosine sim:    {robust_cos:.4f}")
            print(f"  δ L∞ norm:            {info['delta_linf']:.4f}")
            print(f"  δ L2 norm:            {info['delta_l2']:.4f}")
            print(f"{'='*50}")

        return x_adv.detach(), info


class ASPLAttack:
    """
    Alternating Surrogate and Perturbation Learning (from Anti-DreamBooth).

    This is more relevant if we're also trying to protect against DreamBooth
    fine-tuning. For pure face recognition protection, PGDAttack is sufficient.

    The alternation:
    1. Fix δ, update surrogate model (fine-tune on perturbed images)
    2. Fix surrogate, update δ (PGD against the surrogate)
    Repeat T times.

    This gives stronger perturbations because the surrogate adapts to the noise.
    """

    def __init__(
        self,
        face_model: FaceEmbedder,
        eot_wrapper: EoTWrapper,
        epsilon: float = 8 / 255,
        step_size: float = 2 / 255,
        inner_steps: int = 20,        # PGD steps per alternation
        outer_steps: int = 5,         # Number of alternations
        surrogate_lr: float = 1e-5,   # Learning rate for surrogate updates
        surrogate_steps: int = 10,    # Fine-tune steps per alternation
    ):
        self.face_model = face_model
        self.eot = eot_wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.surrogate_lr = surrogate_lr
        self.surrogate_steps = surrogate_steps

    def run(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Run ASPL attack.

        The key difference from plain PGD: we periodically update the
        surrogate model to "see" the current perturbation, making it
        harder for the attack to overfit to a fixed model.
        """
        import copy

        device = x.device
        x = x.detach().clone()

        # Create a surrogate copy that we'll fine-tune
        surrogate = copy.deepcopy(self.face_model)
        surrogate.backbone.train()

        # Unfreeze surrogate for fine-tuning
        for p in surrogate.backbone.parameters():
            p.requires_grad = True

        # Initialize perturbation
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        delta = delta.clamp(0 - x, 1 - x)

        clean_emb = self.face_model(x).detach()

        all_losses = []

        for outer in range(self.outer_steps):
            print(f"\nASPL Outer step {outer + 1}/{self.outer_steps}")

            # Phase 1: Update surrogate on perturbed images
            surrogate.backbone.train()
            opt = torch.optim.Adam(surrogate.backbone.parameters(), lr=self.surrogate_lr)

            for _ in range(self.surrogate_steps):
                x_adv = (x + delta.detach()).clamp(0.0, 1.0)
                emb = surrogate(x_adv)
                # Surrogate tries to match clean embeddings on perturbed inputs
                loss = 1 - F.cosine_similarity(emb, clean_emb, dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Phase 2: Update perturbation against surrogate
            surrogate.backbone.eval()
            for p in surrogate.backbone.parameters():
                p.requires_grad = False

            # Create temporary EoT wrapper with surrogate
            temp_eot = EoTWrapper(
                surrogate,
                num_samples=self.eot.num_samples,
            )

            delta.requires_grad_(True)

            for inner in range(self.inner_steps):
                x_adv = (x + delta).clamp(0.0, 1.0)
                loss = temp_eot(x_adv, clean_emb)
                loss.backward()

                with torch.no_grad():
                    delta.data -= self.step_size * delta.grad.sign()
                    delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                    delta.data = (x + delta.data).clamp(0.0, 1.0) - x
                    all_losses.append(loss.item())

                delta.grad.zero_()

            delta = delta.detach()

            # Re-enable surrogate gradients for next round
            for p in surrogate.backbone.parameters():
                p.requires_grad = True

        x_adv = (x + delta).clamp(0.0, 1.0)

        with torch.no_grad():
            final_emb = self.face_model(x_adv)
            final_cos = F.cosine_similarity(clean_emb, final_emb, dim=1).mean().item()

        info = {
            "loss_history": all_losses,
            "final_cosine_sim": final_cos,
            "outer_steps": self.outer_steps,
            "inner_steps": self.inner_steps,
        }

        return x_adv, info
