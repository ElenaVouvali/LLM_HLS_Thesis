#-----------------------------------------------------------
#                       MAIN
#-----------------------------------------------------------

from config import FLAGS
from train_GNN import train_main, inference
from saver import saver
from utils import load

from os.path import join, exists

import config
TARGETS = config.TARGETS

from data import get_data_list, MyOwnDataset
import data

SAVE_DIR = data.SAVE_DIR


def maybe_load_pragma_dim():
    """
    Load pragma_dim only when needed (mainly for optional GAE-T style paths)
    """
    pragma_dim = None

    candidate = FLAGS.pragma_dim_path
    if candidate is None:
        candidate = join(SAVE_DIR, 'pragma_dim.klepto')

    if FLAGS.gae_T and exists(candidate):
        pragma_dim = load(candidate)
        saver.log_info(f'Loaded pragma_dim from {candidate}')

    return pragma_dim


if __name__ == '__main__':

    # --------------------------------------------------
    # Dataset loading
    # --------------------------------------------------
    if FLAGS.force_regen:
        saver.log_info('Regenerating dataset...')
        dataset, pragma_dim = get_data_list()
    else:
        dataset = MyOwnDataset()
        pragma_dim = maybe_load_pragma_dim()
        saver.log_info(f'Read compact dataset from {SAVE_DIR} with {len(dataset)} samples')

    # --------------------------------------------------
    # Inference helper
    # --------------------------------------------------
    def inf_main(dataset):
        if FLAGS.model_path is None:
            saver.error('model_path must be set for running inference.')
            raise RuntimeError()

        model_paths = FLAGS.model_path if isinstance(FLAGS.model_path, list) else [FLAGS.model_path]

        for ind, model_path in enumerate(model_paths):
            if FLAGS.val_ratio > 0.0:
                inference(
                    dataset,
                    init_pragma_dict=pragma_dim,
                    model_path=model_path,
                    model_id=ind,
                    test_ratio=FLAGS.val_ratio
                )
                inference(
                    dataset,
                    init_pragma_dict=pragma_dim,
                    model_path=model_path,
                    model_id=ind,
                    test_ratio=FLAGS.val_ratio,
                    is_val_set=True
                )

            inference(
                dataset,
                init_pragma_dict=pragma_dim,
                model_path=model_path,
                model_id=ind,
                test_ratio=FLAGS.val_ratio,
                is_train_set=True
            )

            if ind + 1 < len(model_paths):
                saver.new_sub_saver(subdir=f'run{ind+2}')
                saver.log_info('\n\n')

    # --------------------------------------------------
    # Main dispatch
    # --------------------------------------------------
    if FLAGS.subtask == 'inference':
        inf_main(dataset)

    elif FLAGS.subtask == 'train':
        test_ratio, resample_list = FLAGS.val_ratio, [-1]
        if FLAGS.resample:
            test_ratio, resample_list = 0.25, range(4)

        for ind, r in enumerate(resample_list):
            saver.info(f'Starting training with resample {r}')
            train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)

            if ind + 1 < len(resample_list):
                saver.new_sub_saver(subdir=f'run{ind+2}')
                saver.log_info('\n\n')

    else:
        raise NotImplementedError()

    saver.close()
