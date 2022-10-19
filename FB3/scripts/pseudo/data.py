import os, torch, re
from tqdm import tqdm
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset


def read_file(CFG, name):
    with open(f'{CFG.filepath}/data/pseudo/feedback-prize-2021/train/{name}.txt', 'r') as f:
        return re.sub("[\xa0\n ]+", ' ', f.read())

def get_data(CFG):
    train = pd.read_csv(f'{CFG.filepath}/data/train.csv')
    df_train_ex = pd.read_csv(f'{CFG.filepath}/data/pseudo/submission.csv')
    df_train_ex['full_text'] = [read_file(CFG, r.text_id) for i, r in tqdm(df_train_ex.iterrows(), total=len(df_train_ex))]
    train = pd.concat([train, df_train_ex]).reset_index(drop=True)
    if CFG.debug:
        train=train.iloc[5000:5050,:].reset_index(drop=True)
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    return train

def get_tokenizer(CFG):
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(CFG.OUTPUT_DIR+'tokenizer/')
    #CFG.tokenizer = tokenizer
    return tokenizer

def get_maxlen(CFG, train):
    tokenizer = CFG.tokenizer
    lengths = []
    tk0 = tqdm(train['full_text'].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_len = min(max(lengths) + 2, 2048) # cls & sep & sep
    CFG.LOGGER.info(f"max_len: {max_len}")
    return max_len


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs