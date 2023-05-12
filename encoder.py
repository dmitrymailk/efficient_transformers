from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "facebook/xglm-7.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

BATCH = 2
SEQ = 4096
EMB = 2048
VOCAB = len(tokenizer)
HEADS = 32
LAYERS = 48


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    torch_cuda_mem = torch.cuda.mem_get_info(device)
    mem = {
        "used": torch_cuda_mem[-1] - torch_cuda_mem[0],
        "total": torch_cuda_mem[-1],
    }
    print(f"Total GB: {mem['total'] / 1e9} Used GB: {mem['used']/ 1e9}")


encoder_config = {
    "reversible": False,  # Turn on to test the effect of using reversible layers
    "block_type": "encoder",
    "num_layers": LAYERS,
    "dim_model": EMB,
    "residual_norm_style": "post",
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": VOCAB,
    },
    "multi_head_config": {
        "num_heads": HEADS,
        "residual_dropout": 0.1,
        "use_rotary_embeddings": True,
        "attention": {
            "name": "favor",
            "dropout": 0.1,
            "causal": True,
            "seq_len": SEQ,
            "num_rules": 2,
        },
    },
    "feedforward_config": {
        # "name": "MLP",  # Use MLP if Triton is not available
        "name": "FusedMLP",  # Use MLP if Triton is not available
        "dropout": 0.0,
        "activation": "gelu",
        "hidden_layer_multiplier": 4,
    },
}


config = xFormerEncoderConfig(**encoder_config)

device = "cuda:0"
with torch.autocast(device_type="cuda", dtype=torch.float16):
    encoder = xFormerEncoderBlock(config)
    encoder.to(device)
    encoder.to(torch.float16)

    x1 = torch.tensor(
        [[1, 2, 3]],
    ).to(device)
    x1 = torch.randint(1, 6, (BATCH, SEQ)).to(device).to(torch.int)

    y = encoder(x1)
    print_trainable_parameters(encoder)
    print(y.shape, y)

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.to(device)
    # print_trainable_parameters(model)
