import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester
from shutil import copyfile
from config import getConfig
warnings.filterwarnings('ignore')
cfg = getConfig()


def prepare_trained_model_file() -> str:
    trained_model_path = os.path.join(cfg.model_path, cfg.dataset, f"TE{cfg.arch}_0")
    os.makedirs(trained_model_path, exist_ok=True)
    trained_model_file = os.path.join(cfg.model_path, f"copy_model.pth")
    copyfile(trained_model_file, os.path.join(trained_model_path, "best_model.pth"))
    return trained_model_path


def main(cfg):
    print('<---- Training Params ---->')
    pprint.pprint(cfg)

    # Random Seed
    seed = cfg.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # save_path = os.path.join(cfg.model_path)
    save_path = cfg.model_path
    if cfg.action == 'train':
        # Create model directory
        os.makedirs(save_path, exist_ok=True)
        Trainer(cfg, save_path)
    elif cfg.action == 'test':
        datasets = ['car_data']
        for dataset in datasets:
            cfg.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(cfg, save_path).test()

            print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
                  f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')
    elif cfg.action == 'apply':
        # trained_model_path = prepare_trained_model_file()
        # Tester(cfg,"./results/", "model_weights_25.pth", have_gt=False).test()
        for model_name in os.listdir("./results/"):
          if(model_name.split('/')[-1].split('.')[-1]=='pth'):
            print(f'Model ---> {model_name}')
            Tester(cfg, save_path, model_name, have_gt=False).test()
        # print(trained_model_path)
        # input('')
        # shutil.rmtree(trained_model_path)
    else:
        raise ValueError("action should be train, test or apply.")


if __name__ == '__main__':
    main(cfg)
