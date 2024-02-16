from mmcv.runner.hooks import HOOKS, Hook
import torch
from mmcv.runner import Runner
import logging

from typing import Optional
# from mmdet3d.core.hook.utils import is_parallel
from mmcv.utils import get_logger


#@note 用于检查的hook
@HOOKS.register_module()
class CheckHook(Hook):
    '''
    INFO 会输出到文件
    WARNING 会输出到文件和屏幕
    '''
    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False):
        super(CheckHook, self).__init__()
        # self.interval = interval
        # self.ignore_last = ignore_last
        # self.reset_flag = reset_flag
        
        self.start_iter = 0
        #新建一个logger
        self.logger = logging.getLogger('check_hook')
        #loger保存的路径为mmdetection3d/tools/train.py中的work_dir
        self.logger.setLevel(logging.INFO)
        #新建一个handler
        handler = logging.FileHandler('check_hook.log', mode='w')
        handler.setLevel(logging.INFO)
        steamhandle=logging.StreamHandler()
        steamhandle.setLevel(logging.WARNING)
        
        #新建一个formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #将formatter添加到handler中
        handler.setFormatter(formatter)
        steamhandle.setFormatter(formatter)
        #将handler添加到logger中
        self.logger.addHandler(handler)
        self.logger.addHandler(steamhandle)
        self.root_logger = get_logger('mmdet')
        
        
    #记录每次运行前的forward的输出
    def before_run(self, runner: Runner):
        self.logger.info('before_run')
        self.logger.info(f"max_iters:{runner.max_iters}")
        self.logger.info(f"start_iter:{self.start_iter}")
# 13366
        
    def before_train_epoch(self, runner: Runner):
        # self.logger.info('before_train_epoch')
        # 记录每次epoch开始前的状态，用于保留前几次输出
        # if is_parallel(runner.model.module):
        #     runner.model.module.module.iter_dict['iter']=0
        #     runner.model.module.module.iter_dict['epoch']=runner.epoch
            
        # else:
        #     runner.model.module.iter_dict['iter']= 0
        #     runner.model.module.iter_dict['epoch']=runner.epoch
        if hasattr(runner.model.module,"iter_dict"):
            runner.model.module.iter_dict['iter']=0
            runner.model.module.iter_dict['epoch']=runner.epoch
        elif hasattr(runner.model.module.module,"iter_dict"):
            runner.model.module.module.iter_dict['iter']=0
            runner.model.module.module.iter_dict['epoch']=runner.epoch

        
    def after_train_epoch(self, runner: Runner):
        self.logger.info(f'after_train_epoch , epoch:{runner.epoch}')
        
    def before_train_iter(self, runner):
        # self.logger.info('before_train_iter')
        # pass
        self.root_logger.warning(f"before_train_iter, iter:{runner.iter}")
        if hasattr(runner.model.module,"iter_dict"):
            runner.model.module.iter_dict['iter']=runner.inner_iter
        elif hasattr(runner.model.module.module,"iter_dict"):
            runner.model.module.module.iter_dict['iter']=runner.inner_iter
 
        
    def after_train_iter(self,
                         runner: Runner) -> None:
        self.logger.info(f"after_train_iter, iter:{runner.iter}")
        
    def before_val_iter(self, runner):
        # self.logger.info(f'before_val_iter')
        pass
    
    def after_val_iter(self,
                       runner: Runner) -> None:
        self.logger.info(f'after_val_iter, iter:{runner.iter}')

    

    