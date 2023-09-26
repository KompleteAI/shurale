from xllm.core.config import HuggingFaceConfig
from xllm.cli.train import cli_run_train

if __name__ == '__main__':
    cli_run_train(config_cls=HuggingFaceConfig)
