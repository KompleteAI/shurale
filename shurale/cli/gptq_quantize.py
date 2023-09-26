from xllm.core.config import HuggingFaceConfig
from xllm.cli.gptq_quantize import cli_run_gptq_quantize

if __name__ == '__main__':
    cli_run_gptq_quantize(config_cls=HuggingFaceConfig)
