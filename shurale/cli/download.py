from xllm.core.config import HuggingFaceConfig
from xllm.cli.download import cli_run_download

if __name__ == "__main__":
    cli_run_download(config_cls=HuggingFaceConfig)
