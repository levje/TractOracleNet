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

def score_tractograms(reference, tractograms, scoring_data_dir, out_dir, scored_extension="trk", is_ismrm=False):

    reference = Path(reference)
    scoring_data_dir = Path(scoring_data_dir)
    out_dir = Path(out_dir)

    assert scoring_data_dir.exists(), f"Scoring data folder {scoring_data_dir} does not exist."
    assert reference.exists(), f"Reference {reference} does not exist."

    for tractogram in tractograms:

        tractogram_path = Path(tractogram)
        assert tractogram_path.exists(), f"Tractogram {tractogram} does not exist."

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            if is_ismrm:
                scoring_args = [
                    "./scil_score_ismrm_Renauld2023.sh",
                    tractogram_path, # in_tractogram
                    tmp_path, # out_dir
                    scoring_data_dir, # scoring_data 
                ]
            else:
                gt_config = os.path.join(scoring_data_dir, "scil_scoring_config.json")
                if not os.path.exists(gt_config):
                    raise FileNotFoundError(f"There should be a ground truth config file at {gt_config}, but it doesn't exist.")

                scoring_args = [
                    "scil_tractogram_segment_and_score.py",
                    tractogram_path, # in_tractogram
                    gt_config, # gt_config
                    tmp_path, # out_dir
                    "--reference", reference, # reference
                ]
            

            # Segment and score the tractogram
            c_proc = subprocess.run([*scoring_args])
            c_proc.check_returncode() # Throws if the process failed

            merge_with_scores(reference, tmp_path / "segmented_VB", tmp_path / "IS.trk", out_dir / "scored_{}.{}".format(tractogram_path.stem, scored_extension))


if '__main__' == __name__:

    # Important, this script is intended to be ran from the TractOracleNet/TractOracleNet/dataset directory.
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=str, help='Path to the reference streamlines file (e.g. FA map).')
    parser.add_argument('scoring_data_dir', type=str, help='Path to the scoring data folder.')
    parser.add_argument('tractograms', nargs='+', type=str, help='Path to the tractogram files to score.')
    parser.add_argument('--out_dir', type=str, help='Output directory to save the scored tractograms.', default='.')
    parser.add_argument('--is_ismrm', action='store_true', help='Flag to indicate if the dataset is ISMRM2015. If so, use the updated scoring script.', default=False)

    args = parser.parse_args()
    reference = args.reference
    scoring_data_dir = args.scoring_data_dir
    tractograms = args.tractograms
    out_dir = args.out_dir

    score_tractograms(reference, tractograms, scoring_data_dir, out_dir, scored_extension="trk", is_ismrm=is_ismrm)