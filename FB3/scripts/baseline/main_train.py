from config import *
from utils import *
from data import *
from train import *

if __name__ == '__main__':
    CFG.LOGGER = get_logger(CFG)
    CFG.tokenizer = get_tokenizer(CFG)
    train = get_data(CFG)
    CFG.max_len = get_maxlen(CFG, train)
    wanb_init(CFG)

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(CFG, train, fold, test=False)
                oof_df = pd.concat([oof_df, _oof_df])
                CFG.LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(CFG, _oof_df)
        oof_df = oof_df.reset_index(drop=True)
        CFG.LOGGER.info(f"========== CV ==========")
        get_result(CFG, oof_df)
        oof_df.to_pickle(CFG.OUTPUT_DIR + 'oof_df.pkl')

    if CFG.wandb:
        wandb.finish()