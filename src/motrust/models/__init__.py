"""Representation-learning components used by MoTRUST."""

from .batch_discriminator import BatchDiscriminator
from .consistency_loss import ConsistencyLossModule, compute_consistency_loss
from .latent_stability import AdaptiveStabilityConstraint, LatentStabilityController
from .mosaic_regularizers import (
    confidence_weighted_transport_pull,
    soft_neighborhood_preservation_loss,
    symmetric_confidence_transport_loss,
    transport_correspondence_confidence,
)
from .multimodal_vae import ImprovedMultiModalVAE
from .poe import ProductOfExperts, poe_fusion
from .unbalanced_ot import UncertaintyAwareUnbalancedOT

MoTRUSTVAE = ImprovedMultiModalVAE

__all__ = [
    "AdaptiveStabilityConstraint",
    "BatchDiscriminator",
    "ConsistencyLossModule",
    "ImprovedMultiModalVAE",
    "LatentStabilityController",
    "MoTRUSTVAE",
    "ProductOfExperts",
    "UncertaintyAwareUnbalancedOT",
    "compute_consistency_loss",
    "confidence_weighted_transport_pull",
    "poe_fusion",
    "soft_neighborhood_preservation_loss",
    "symmetric_confidence_transport_loss",
    "transport_correspondence_confidence",
]
