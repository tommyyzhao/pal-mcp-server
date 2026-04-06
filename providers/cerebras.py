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

    # Canonical model identifiers used for category routing.
    PRIMARY_MODEL = "zai-glm-4.7"
    FALLBACK_MODEL = "zai-glm-4.7"

    def __init__(self, api_key: str, **kwargs):
        """Initialize Cerebras provider with API key."""
        # Set Cerebras base URL
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
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer zai-glm-4.7 for advanced reasoning tasks
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # zai-glm-4.7 is the only model, use it
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        else:  # BALANCED or default
            # zai-glm-4.7 is the only model, use it
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]


# Load registry data at import time
CerebrasModelProvider._ensure_registry()
