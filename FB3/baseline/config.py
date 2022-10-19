import torch

class CFG:
    OUTPUT_DIR = '/cluster/home/u27/FB3/output/'
    filepath = '/cluster/home/u27/FB3'
    wandb = False
    competition = 'FB3'
    _wandb_kernel = 'nakama'
    debug = True
    apex = True
    print_freq = 1
    num_workers = 0
    model = "microsoft/deberta-v3-large"
    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 4
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 4
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    train = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]