from xllm.core.config import HuggingFaceConfig
from xllm.cli.fuse import cli_run_fuse

if __name__ == "__main__":
    cli_run_fuse(config_cls=HuggingFaceConfig)
