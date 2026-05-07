import importlib

def prepare_attn_implementation():
    try:
        importlib.import_module("flash_attn")
        return "flash_attention_2"
    except Exception as exc:
        print(f"[train_mem] flash_attention_2 unavailable, fallback to default attention: {exc}")
        import transformers.utils.import_utils as import_utils

        original_is_package_available = import_utils._is_package_available

        def patched_is_package_available(pkg_name, *args, **kwargs):
            if pkg_name == "flash_attn":
                return False
            return original_is_package_available(pkg_name, *args, **kwargs)

        import_utils._is_package_available = patched_is_package_available
        return None

if __name__ == "__main__":
    attn_implementation = prepare_attn_implementation()
    from llava.train.train import train

    train(attn_implementation=attn_implementation)
