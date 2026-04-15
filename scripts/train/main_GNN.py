#-----------------------------------------------------------
#                       MAIN
#-----------------------------------------------------------


from config import FLAGS
from train_GNN import train_main, inference
from saver import saver
from utils import get_root_path, load

from os.path import join, dirname
from glob import iglob

import config
TARGETS = config.TARGETS
from data import get_data_list, MyOwnDataset
import data


if __name__ == '__main__':

    if not FLAGS.force_regen:
        try:
            with open("good_files.txt") as f:
                good_files = [line.strip() for line in f if line.strip()]
            dataset = MyOwnDataset(data_files=good_files)
            print(f"read filtered dataset with {len(dataset)} graphs")
        except FileNotFoundError:
            # fallback: all graphs
            dataset = MyOwnDataset()
            print('read dataset (all graphs)')
        pragma_dim = None

    else:
        pragma_dim = 0
        dataset, pragma_dim = get_data_list()

    if FLAGS.encoder_path is not None:
#        pragma_dim = load(join(dirname(FLAGS.encoder_path), 'v18_pragma_dim'))
        pragma_dim = load(FLAGS.pragma_dim_path)


    def inf_main(dataset):
        if type(FLAGS.model_path) is None:
            saver.error('model_path must be set for running the inference.')
            raise RuntimeError()
        else:
            for ind, model_path in enumerate(FLAGS.model_path):
                if FLAGS.val_ratio > 0.0:
                    inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio)
                    inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_val_set=True)
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_train_set=True)
                if ind + 1 < len(FLAGS.model_path):
                    saver.new_sub_saver(subdir=f'run{ind+2}')
                    saver.log_info('\n\n')


    if FLAGS.subtask == 'inference':
        inf_main(dataset)

    elif FLAGS.subtask == 'train':
        test_ratio, resample_list = FLAGS.val_ratio, [-1]
        if FLAGS.resample:
            test_ratio, resample_list = 0.25, range(4)
        for ind, r in enumerate(resample_list):
            saver.info(f'Starting training with resample {r}')
            test_data = train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)
            if ind + 1 < len(resample_list):
                saver.new_sub_saver(subdir=f'run{ind+2}')
                saver.log_info('\n\n')

    else:
        raise NotImplementedError()

    saver.close()
