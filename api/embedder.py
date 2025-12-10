import adalflow as adal
import logging

from api.config import configs, get_embedder_type, get_embedder_config

logger = logging.getLogger(__name__)


def get_embedder(is_local_ollama: bool = False, use_google_embedder: bool = False,
                 embedder_type: str = None) -> adal.Embedder:
    """Get embedder based on configuration or parameters.

    Args:
        is_local_ollama: Legacy parameter for Ollama embedder
        use_google_embedder: Legacy parameter for Google embedder
        embedder_type: Direct specification of embedder type ('ollama', 'google', 'openai')

    Returns:
        adal.Embedder: Configured embedder instance
    """
    # Determine which embedder config to use
    if embedder_type:
        # Use the specified embedder type
        if embedder_type == 'google' and 'embedder_google' in configs:
            embedder_config = configs.get("embedder_google", {})
        elif embedder_type == 'ollama' and 'embedder_ollama' in configs:
            embedder_config = configs.get("embedder_ollama", {})
        else:
            embedder_config = configs.get("embedder", {})
    else:
        # Use the default embedder config from environment
        embedder_config = get_embedder_config()
    
    # Check if embedder_config is empty or missing required keys
    if not embedder_config:
        raise ValueError("Embedder configuration not found. Please ensure embedder configuration files exist or set DEEPWIKI_EMBEDDER_TYPE environment variable.")
    
    if "model_client" not in embedder_config:
        raise ValueError("model_client not found in embedder configuration")
    
    if "model_kwargs" not in embedder_config:
        raise ValueError("model_kwargs not found in embedder configuration")

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    # Create embedder with basic parameters
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}

    embedder = adal.Embedder(**embedder_kwargs)

    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder