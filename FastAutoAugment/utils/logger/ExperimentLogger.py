import os
import time
import torch
import shutil
import datetime

from torch.utils.tensorboard import SummaryWriter


class Log:
    def __init__(self, root, is_print=True):
        self.root = root
        self.is_print = is_print

    def write(self, string, end='\n'):
        datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        string = '%s: %s' % (datetime_string, string)
        if self.is_print:
            print(string, end=end)
        with open(self.root, 'a') as f:
            f.write(string + end)


class ExperimentLogger:
    def __init__(self, save_root, exist_ok=False, log_file='log.log'):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=exist_ok)
        self.log = Log(os.path.join(self.save_root, log_file), is_print=True)
        self.tensor_log = None
        os.makedirs(os.path.join(self.save_root, 'checkpoint'), exist_ok=exist_ok)
        self.start_time = time.time()

    def save_args(self, args):
        args_root = os.path.join(self.save_root, 'train_args.txt')
        with open(args_root, 'w') as f:
            f.write('python3 train.py \\\n')
            for arg in args.__dict__.keys():
                f.write('--%s %s \\\n' % (arg, args.__dict__[arg]))
        self.log.write(str(args))

    def save_yaml(self, cfg):
        yaml_root = os.path.join(self.save_root, 'config.yaml')
        with open(yaml_root, 'w') as f:
            f.write(cfg.dump())
        self.log.write(cfg.dump())

    def save_src(self, src_root):
        shutil.copytree(src_root, os.path.join(self.save_root, 'src'))
        self.log.write('Save src')

    def save_checkpoint(self, checkpoint, tag):
        model_root = os.path.join(self.save_root, 'checkpoint', '%s.pth' % str(tag))
        torch.save(checkpoint, model_root)
        self.log.write('Checkpoint save to %s' % model_root)

    def load_checkpoint(self, tag):
        model_root = os.path.join(self.save_root, 'checkpoint', '%s.pth' % str(tag))
        checkpoint = torch.load(model_root)
        self.log.write('Checkpoint load to %s' % model_root)
        return checkpoint

    def log_model_architecture(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        self.log.write('---------- Networks architecture -------------')
        self.log.write(str(model))
        self.log.write('Total number of parameters: %d' % num_params)
        self.log.write('----------------------------------------------')

    def log_dataset(self, dataset):
        self.log.write('%s %s Dataset [Sampling: %s]' % (dataset.domain, dataset.partition, dataset.sampling_method))
        self.log.write(str(dataset.get_info()))
        self.log.write('Load %d Data' % len(dataset))

    def log_epoch(self, epoch, total_epoch):
        time_per_epoch = (time.time() - self.start_time) / (epoch + 1)
        total_eta = (total_epoch - (epoch + 1)) * time_per_epoch
        self.log.write('Epoch %d Complete | time per epoch %s / total eta %s' %
                       (epoch, sec_to_time(time_per_epoch), sec_to_time(total_eta)))

    def log_classification(self, partition, loss, class_accuracy, sample_accuracy, accuracy_per_class):
        self.log.write('%s Loss: %.4lf' % (partition, loss))
        self.log.write('%s class accuracy: %.4lf / %s sample accuracy: %.4lf' %
                       (partition, class_accuracy, partition, sample_accuracy))
        self.log.write('%s accuracy per class' % partition)
        self.log.write(str([('%.4lf' % acc) for acc in accuracy_per_class]))

    def log_tensorlog(self, epoch, log_dict):
        if self.tensor_log is None:
            self.tensor_log = SummaryWriter(self.save_root)

        for key in log_dict:
            self.tensor_log.add_scalar(key, log_dict[key], epoch)

    def change_log_file(self, log_file):
        self.log = Log(os.path.join(self.save_root, log_file), is_print=True)


def sec_to_time(sec):
    sec = int(sec)
    hour = sec // 3600
    min = (sec - (hour * 3600)) // 60
    sec = sec - (hour * 3600) - (min * 60)

    return '%d:%d:%d' % (hour, min, sec)
