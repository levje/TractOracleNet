import argparse
import tempfile
import numpy as np
import subprocess

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path

def merge_with_scores(reference, bundles_dir, inv_streamlines, output):

    file_list = os.listdir(bundles_dir)
    main_tractogram = None

    # Add valid streamlines
    for file in file_list:
        tractogram = load_tractogram(os.path.join(bundles_dir, file), str(reference))
        num_streamlines = len(tractogram.streamlines)

        tractogram.data_per_streamline['score'] = np.ones(num_streamlines, dtype=np.float32)

        if main_tractogram is None:
            main_tractogram = tractogram
        else:
            main_tractogram = main_tractogram + tractogram

    assert inv_streamlines.exists(), f"Invalid streamlines {inv_streamlines} does not exist."
    assert Path(reference).exists(), f"Reference {reference} does not exist."

    # Add invalid streamlines
    inv_tractogram = load_tractogram(str(inv_streamlines), str(reference))
    num_streamlines = len(inv_tractogram.streamlines)
    inv_tractogram.data_per_streamline['score'] = np.zeros(num_streamlines, dtype=np.float32)

    main_tractogram = main_tractogram + inv_tractogram

    assert main_tractogram is not None, "No valid streamlines found."
    
    print(f"Main tractogram has {len(main_tractogram.streamlines)} streamlines.")
    print(f"Saving tractogram to {output}")
    save_tractogram(main_tractogram, str(output))

def score_tractograms(reference, tractograms, gt_config, out_dir, scored_extension="trk"):

    reference = Path(reference)
    gt_config = Path(gt_config)
    out_dir = Path(out_dir)

    assert gt_config.exists(), f"Ground truth config {gt_config} does not exist."
    assert reference.exists(), f"Reference {reference} does not exist."

    for tractogram in tractograms:

        tractogram_path = Path(tractogram)
        assert tractogram_path.exists(), f"Tractogram {tractogram} does not exist."

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            scoring_args = [
                tractogram_path, # in_tractogram
                gt_config, # gt_config
                tmp_path, # out_dir
                "--reference", reference
            ]
            c_proc = subprocess.run(["pwd"])
            # Segment and score the tractogram
            c_proc = subprocess.run(["scil_tractogram_segment_and_score.py", *scoring_args])
            c_proc.check_returncode() # Throws if the process failed

            merge_with_scores(reference, tmp_path / "segmented_VB", tmp_path / "IS.trk", out_dir / "scored_{}.{}".format(tractogram_path.stem, scored_extension))


if '__main__' == __name__:

    # Important, this script is intended to be ran from the TractOracleNet/TractOracleNet/dataset directory.

    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=str, help='Path to the reference streamlines file (e.g. FA map).')
    parser.add_argument('gt_config', type=str, help='Path to the ground truth config file.')
    parser.add_argument('tractograms', nargs='+', type=str, help='Path to the tractogram files to score.')
    parser.add_argument('--out_dir', type=str, help='Output directory to save the scored tractograms.', default='.')

    args = parser.parse_args()
    reference = args.reference
    gt_config = args.gt_config
    tractograms = args.tractograms
    out_dir = args.out_dir

    score_tractograms(reference, tractograms, gt_config, out_dir, scored_extension="trk")