"""Shared model selection utilities for all scripts."""

AVAILABLE_MODELS = {
    '1d_cnn': ('1d_cnn', 'Anomaly1DCNN'),
    'resnet': ('resNet', 'ResNet1D'),
    'cnn_attention': ('cnn_attention', 'CNNAttention'),
}

MODEL_DISPLAY_NAMES = {
    '1d_cnn': '1D-CNN',
    'resnet': 'ResNet 1D',
    'cnn_attention': 'CNN+Attention',
}


def get_model(model_name: str, num_classes: int, in_channels: int = 1):
    """Dynamically import and instantiate the selected model."""
    import importlib

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS.keys())}")

    module_name, class_name = AVAILABLE_MODELS[model_name]
    mod = importlib.import_module(module_name)
    model_class = getattr(mod, class_name)

    if model_name == '1d_cnn':
        return model_class(in_channels=in_channels, num_classes=num_classes)
    else:
        return model_class(num_classes=num_classes, in_channels=in_channels)


def select_model() -> str:
    """Interactive model selection. Returns the model key string."""
    import sys

    # Check if --model was passed as CLI argument
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            choice = sys.argv[idx + 1]
            if choice in AVAILABLE_MODELS:
                return choice
            else:
                print(f"Unknown model '{choice}'.")

    # Interactive prompt
    keys = list(AVAILABLE_MODELS.keys())
    print("\n" + "=" * 40)
    print("  SELECT MODEL ARCHITECTURE")
    print("=" * 40)
    for i, key in enumerate(keys, 1):
        print(f"  [{i}] {MODEL_DISPLAY_NAMES[key]}")
    print("=" * 40)

    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                selected = keys[idx]
                print(f"→ Selected: {MODEL_DISPLAY_NAMES[selected]}\n")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"  Invalid choice. Please enter 1-{len(keys)}.")
