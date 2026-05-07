import importlib


def disable_broken_flash_attn():
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception as exc:
        print(f"[llava] flash_attention_2 unavailable, fallback to default attention: {exc}")
        import transformers.utils.import_utils as import_utils

        original_is_package_available = import_utils._is_package_available

        def patched_is_package_available(pkg_name, *args, **kwargs):
            if pkg_name == "flash_attn":
                return False
            return original_is_package_available(pkg_name, *args, **kwargs)

        import_utils._is_package_available = patched_is_package_available
        return False
