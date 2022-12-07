import torch
import numpy as np
import random
import os

# random seed 환경 설정
def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.default_rng(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    
# DataLoader는 다중 프로세스 데이터 로드 알고리즘에서 임의성에 따라 작업자를 다시 시드합니다. 
# 재현성을 유지하기 위한 사용 worker_init_fn()및 생성기
def seed_worker(random_seed):

    g = torch.Generator()
    g.manual_seed(random_seed)
    return g