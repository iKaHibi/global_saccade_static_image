import os

# --- 必须放在任何其他 import 之前 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import main

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run simulation pipeline with configurable config path.")

    # Add optional positional argument
    parser.add_argument(
        "cfg_path_str",
        nargs="?",  # '?' makes the argument optional
        # default="configs/fig2_rf_size_config_large_saccade.json",
        default="configs/fig2_rf_size_config_drosophila_fem.json",
        help="Path to the configuration JSON file"
    )

    parser.add_argument(
        "num_samples",
        nargs="?",  # '?' makes the argument optional
        default=12,
        type=int,
        help="number of samples to use for simulation"
    )

    parser.add_argument(
        "only_shift",
        nargs="?",
        default=True,
        type=str2bool,
        help="Enable shift only (True/False)"
    )

    args = parser.parse_args()

    # Pass the argument (or the default) to main
    main(args.cfg_path_str, n_samples=args.num_samples, split_by_jitter=args.only_shift)