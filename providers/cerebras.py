"""Cerebras Inference model provider implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.cerebras import CerebrasModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class CerebrasModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for Cerebras Inference API (ZAI-GLM models).

    Publishes capability metadata for the officially supported deployments and
    maps tool-category preferences to the appropriate Cerebras model.
    """

    FRIENDLY_NAME = "Cerebras"

    REGISTRY_CLASS = CerebrasModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Canonical model identifier — single-model provider for now.
    PRIMARY_MODEL = "zai-glm-4.7"

    def __init__(self, api_key: str, **kwargs):
        """Initialize Cerebras provider with API key."""
        kwargs.setdefault("base_url", "https://api.cerebras.ai/v1")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.CEREBRAS

    def get_preferred_model(self, category: ToolModelCategory, allowed_models: list[str]) -> str | None:
        """Get Cerebras's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        if not allowed_models:
            return None

        # Single-model provider: return PRIMARY_MODEL if allowed, else first available.
        if self.PRIMARY_MODEL in allowed_models:
            return self.PRIMARY_MODEL
        return allowed_models[0]


# Load registry data at import time
CerebrasModelProvider._ensure_registry()
