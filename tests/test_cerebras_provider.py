"""Tests for Cerebras provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.cerebras import CerebrasModelProvider
from providers.shared import ProviderType


class TestCerebrasProvider:
    """Test Cerebras provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = CerebrasModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.CEREBRAS
        assert provider.base_url == "https://api.cerebras.ai/v1"

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = CerebrasModelProvider("test-key", base_url="https://custom.cerebras.ai/v1")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.cerebras.ai/v1"

    def test_model_validation(self):
        """Test model name validation."""
        provider = CerebrasModelProvider("test-key")

        # Test valid models
        assert provider.validate_model_name("zai-glm-4.7") is True
        assert provider.validate_model_name("cerebras") is True
        assert provider.validate_model_name("glm") is True
        assert provider.validate_model_name("glm-4.7") is True
        assert provider.validate_model_name("zai") is True
        assert provider.validate_model_name("zai-glm") is True

        # Test invalid model
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False
        assert provider.validate_model_name("grok-4") is False

    def test_resolve_model_name(self):
        """Test model name resolution."""
        provider = CerebrasModelProvider("test-key")

        # Test shorthand resolution
        assert provider._resolve_model_name("cerebras") == "zai-glm-4.7"
        assert provider._resolve_model_name("glm") == "zai-glm-4.7"
        assert provider._resolve_model_name("glm-4.7") == "zai-glm-4.7"
        assert provider._resolve_model_name("zai") == "zai-glm-4.7"
        assert provider._resolve_model_name("zai-glm") == "zai-glm-4.7"

        # Test full name passthrough
        assert provider._resolve_model_name("zai-glm-4.7") == "zai-glm-4.7"

    def test_get_capabilities(self):
        """Test getting model capabilities for zai-glm-4.7."""
        provider = CerebrasModelProvider("test-key")

        capabilities = provider.get_capabilities("zai-glm-4.7")
        assert capabilities.model_name == "zai-glm-4.7"
        assert capabilities.friendly_name == "Cerebras (zai-glm-4.7)"
        assert capabilities.context_window == 131072
        assert capabilities.max_output_tokens == 40000
        assert capabilities.provider == ProviderType.CEREBRAS
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is False
        assert capabilities.supports_temperature is True

        # Test temperature range
        assert capabilities.temperature_constraint.min_temp == 0.0
        assert capabilities.temperature_constraint.max_temp == 2.0
        assert capabilities.temperature_constraint.default_temp == 0.3

    def test_get_capabilities_with_shorthand(self):
        """Test getting model capabilities with shorthand."""
        provider = CerebrasModelProvider("test-key")

        capabilities = provider.get_capabilities("cerebras")
        assert capabilities.model_name == "zai-glm-4.7"  # Should resolve to full name
        assert capabilities.context_window == 131072

        capabilities_glm = provider.get_capabilities("glm")
        assert capabilities_glm.model_name == "zai-glm-4.7"

    def test_unsupported_model_capabilities(self):
        """Test error handling for unsupported models."""
        provider = CerebrasModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider cerebras"):
            provider.get_capabilities("invalid-model")

    def test_extended_thinking_flags(self):
        """Cerebras capabilities should expose extended thinking support correctly."""
        provider = CerebrasModelProvider("test-key")

        thinking_aliases = [
            "zai-glm-4.7",
            "cerebras",
            "glm",
            "glm-4.7",
            "zai",
            "zai-glm",
        ]
        for alias in thinking_aliases:
            assert provider.get_capabilities(alias).supports_extended_thinking is True

    def test_provider_type(self):
        """Test provider type identification."""
        provider = CerebrasModelProvider("test-key")
        assert provider.get_provider_type() == ProviderType.CEREBRAS

    @patch.dict(os.environ, {"CEREBRAS_ALLOWED_MODELS": "zai-glm-4.7"})
    def test_model_restrictions(self):
        """Test model restrictions functionality."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = CerebrasModelProvider("test-key")

        # zai-glm-4.7 should be allowed (including alias)
        assert provider.validate_model_name("zai-glm-4.7") is True
        assert provider.validate_model_name("cerebras") is True

    @patch.dict(os.environ, {"CEREBRAS_ALLOWED_MODELS": "cerebras"})
    def test_multiple_model_restrictions(self):
        """Restrictions should allow aliases for Cerebras."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = CerebrasModelProvider("test-key")

        # Alias should be allowed (resolves to zai-glm-4.7)
        assert provider.validate_model_name("cerebras") is True
        assert provider.validate_model_name("zai-glm-4.7") is True

    @patch.dict(os.environ, {"CEREBRAS_ALLOWED_MODELS": "zai-glm-4.7,cerebras,glm"})
    def test_both_shorthand_and_full_name_allowed(self):
        """Test that aliases and canonical names can be allowed together."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = CerebrasModelProvider("test-key")

        # Both shorthand and full name should be allowed when explicitly listed
        assert provider.validate_model_name("zai-glm-4.7") is True
        assert provider.validate_model_name("cerebras") is True
        assert provider.validate_model_name("glm") is True

    @patch.dict(os.environ, {"CEREBRAS_ALLOWED_MODELS": ""})
    def test_empty_restrictions_allows_all(self):
        """Test that empty restrictions allow all models."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = CerebrasModelProvider("test-key")

        assert provider.validate_model_name("zai-glm-4.7") is True
        assert provider.validate_model_name("cerebras") is True
        assert provider.validate_model_name("glm") is True

    def test_friendly_name(self):
        """Test friendly name constant."""
        provider = CerebrasModelProvider("test-key")
        assert provider.FRIENDLY_NAME == "Cerebras"

        capabilities = provider.get_capabilities("zai-glm-4.7")
        assert capabilities.friendly_name == "Cerebras (zai-glm-4.7)"

    def test_supported_models_structure(self):
        """Test that MODEL_CAPABILITIES has the correct structure."""
        provider = CerebrasModelProvider("test-key")

        # Check that all expected base models are present
        assert "zai-glm-4.7" in provider.MODEL_CAPABILITIES

        # Check model configs have required fields
        from providers.shared import ModelCapabilities

        config = provider.MODEL_CAPABILITIES["zai-glm-4.7"]
        assert isinstance(config, ModelCapabilities)
        assert hasattr(config, "context_window")
        assert hasattr(config, "supports_extended_thinking")
        assert hasattr(config, "aliases")
        assert config.context_window == 131072
        assert config.supports_extended_thinking is True
        assert config.max_output_tokens == 40000

        # Check aliases are correctly structured
        assert "cerebras" in config.aliases
        assert "glm" in config.aliases
        assert "glm-4.7" in config.aliases
        assert "zai" in config.aliases
        assert "zai-glm" in config.aliases

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        """Test that generate_content resolves aliases before making API calls.

        This is the CRITICAL test that ensures aliases like 'cerebras' get resolved
        to 'zai-glm-4.7' before being sent to Cerebras API.
        """
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "zai-glm-4.7"  # API returns the resolved model name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = CerebrasModelProvider("test-key")

        # Call generate_content with alias 'cerebras'
        result = provider.generate_content(
            prompt="Test prompt",
            model_name="cerebras",
            temperature=0.7,  # This should be resolved to "zai-glm-4.7"
        )

        # Verify the API was called with the RESOLVED model name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive "zai-glm-4.7", not "cerebras"
        assert (
            call_kwargs["model"] == "zai-glm-4.7"
        ), f"Expected 'zai-glm-4.7' but API received '{call_kwargs['model']}'"

        # Verify other parameters
        assert call_kwargs["temperature"] == 0.7
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

        # Verify response
        assert result.content == "Test response"
        assert result.model_name == "zai-glm-4.7"  # Should be the resolved name

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_other_aliases(self, mock_openai_class):
        """Test other alias resolutions in generate_content."""
        from unittest.mock import MagicMock

        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = CerebrasModelProvider("test-key")

        # Test glm -> zai-glm-4.7
        mock_response.model = "zai-glm-4.7"
        provider.generate_content(prompt="Test", model_name="glm", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "zai-glm-4.7"

        # Test glm-4.7 -> zai-glm-4.7
        provider.generate_content(prompt="Test", model_name="glm-4.7", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "zai-glm-4.7"

        # Test zai -> zai-glm-4.7
        provider.generate_content(prompt="Test", model_name="zai", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "zai-glm-4.7"

        # Test zai-glm -> zai-glm-4.7
        provider.generate_content(prompt="Test", model_name="zai-glm", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "zai-glm-4.7"

        # Test zai-glm-4.7 -> zai-glm-4.7 (passthrough)
        provider.generate_content(prompt="Test", model_name="zai-glm-4.7", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "zai-glm-4.7"
