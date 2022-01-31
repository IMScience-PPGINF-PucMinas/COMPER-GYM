from data import logger as train_logger
from data import trial_logger as  trial_logger
from data import eval_logger as eval_logger
from datetime import datetime
import os



train_logger.session("./log/train/").__enter__()
eval_logger.session("./log/eval/").__enter__()
trial_logger.session("./log/trials/").__enter__()


def log(logger_obj,log_data_dict):
    for k, v in log_data_dict:
        logger_obj.logkv(k, v)
    logger_obj.dumpkvs()

def log_train_data(trial,log_itr,task_name,dt_string,tm_len,rtm_len,ep,itr,done,episodic_reward,episodic_reward_itr,
                last_ep_avg_reward, avg_trial_rew):
    log_data_dict =[('Trial',trial),
                    ('Count',log_itr),
                    ('Task',task_name),
                    ('Time',dt_string),
                    ('TMCount',tm_len),
                    ('RTMCount',rtm_len),
                    ('Ep', ep),
                    ('Itr', itr),
                    ("Done",done),
                    ('Rew', episodic_reward),
                    ('AvgEpRew', episodic_reward_itr),
                    ('AvgLasEpRew', last_ep_avg_reward),
                    ('AvgTrialRew', avg_trial_rew)]

    log(train_logger,log_data_dict)

def log_eval_data(trial,dt_string,task_name,iterations,avg_trial_rew):
    log_data_dict =[
        ('Trial',trial),
        ('Time',dt_string),
        ('Task',task_name),
        ('Int',iterations),
        ('EvalAvgRew',avg_trial_rew)]
    log(eval_logger,log_data_dict)

def log_trial_data(trial,dt_string,tep,tqt,lstmep,start,upcritic,bs):
    log_data_dict =[('Trial',trial),
                    ('Time',dt_string),
                    ('TotalEp',tep),
                    ('Tqt',tqt),
                    ('Lstmep',lstmep),
                    ('StartLearn',start),
                    ('Upcritic',upcritic),
                    ('Qlstm_bs',bs)]
    log(trial_logger,log_data_dict)
    