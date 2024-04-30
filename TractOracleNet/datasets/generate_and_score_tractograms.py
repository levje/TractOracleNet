import subprocess
import argparse
from pathlib import Path
from score_tractogram import score_tractograms


def validate_input(dataset, tractogram_extension, scored_extension):
    if dataset not in ["fibercup", "ismrm2015"]:
        raise ValueError("Invalid dataset. Please choose either 'fibercup' or 'ismrm2015'.")
    if tractogram_extension not in ["trk", "tck"]:
        raise ValueError("Invalid extension. Please choose either 'trk' or 'tck' for the tractogram extension.")
    if scored_extension not in ["trk", "tck"]:
        raise ValueError("Invalid extension. Please choose either 'trk' or 'tck'.")


def scil_tracking_local(algo, dataset, out_filepath, npv):
    subprocess.run([
        "scil_tracking_local.py",
        "--algo={}".format(algo),
        "--npv={}".format(npv),
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "{}/masks/{}_mask_wm.nii.gz".format(dataset, dataset),
        str(out_filepath),
    ])


def scil_tracking_pft(algo, dataset, out_filepath, npv):
    subprocess.run([
        "scil_tracking_pft.py",
        "--algo={}".format(algo),
        "--npv={}".format(npv),
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "{}/maps/{}_map_include.nii.gz".format(dataset, dataset),
        "{}/maps/{}_map_exclude.nii.gz".format(dataset, dataset),
        str(out_filepath),
    ])


def main(args):
    dataset = args.dataset
    tractogram_extension = args.extension
    scored_extension = args.scored_extension
    npv = args.npv

    validate_input(dataset, tractogram_extension, scored_extension)

    tractograms_directory = Path("{}/tractograms".format(dataset))
    tractograms_filepaths = []

    # Make sure the directory is empty if it already exists
    if tractograms_directory.exists():
        # Assert the directory is empty
        assert not any(tractograms_directory.iterdir()), "Tractograms directory is not empty."
    else:
        tractograms_directory.mkdir()

    # Local tracking
    scil_tracking_local_algos = ["det", "prob", "eudx"]
    for algo in scil_tracking_local_algos:
        out_filename = "{}_local_{}.{}".format(dataset, algo, tractogram_extension)
        out_filepath = tractograms_directory / out_filename
        
        print("Tracking with local {}".format(algo))
        scil_tracking_local(algo, dataset, out_filepath, npv=npv)

        tractograms_filepaths.append(out_filepath)

    # PFT tracking
    scil_tracking_pft_algos = ["det", "prob"]
    for algo in scil_tracking_pft_algos:
        out_filename = "{}_pft_{}.{}".format(dataset, algo, tractogram_extension)
        out_filepath = tractograms_directory / out_filename

        print("Tracking with pft {}".format(algo))
        scil_tracking_pft(algo, dataset, out_filepath, npv=npv)

        tractograms_filepaths.append(out_filepath)

    # Make sure the scoring directory exists
    scored_tractograms_directory = tractograms_directory / "scored_tractograms"
    if not scored_tractograms_directory.exists():
        scored_tractograms_directory.mkdir()

    # Score tractograms using tractometer, then fusing the scores of the filtered/unfiltered streamlines into a single tractogram.
    score_tractograms(
        "{}/anat/{}_T1.nii.gz".format(dataset, dataset),            # reference
        tractograms_filepaths,                                      # tractograms to score
        "{}/scoring_data/scil_scoring_config.json".format(dataset), # gt_config
        "{}/tractograms/scored_tractograms/".format(dataset),       # out_dir
        scored_extension=scored_extension
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and score tractograms for a given dataset.")
    parser.add_argument("dataset", type=str, default="fibercup", help="Dataset to use for generating and scoring tractograms.", choices=["fibercup", "ismrm2015"])
    parser.add_argument("--extension", type=str, default="tck", help="Extension to use for the generated tractograms.", choices=["trk", "tck"])
    parser.add_argument("--scored_extension", type=str, default="trk", help="Extension to use for the scored tractograms.", choices=["trk", "tck"])
    parser.add_argument("--npv", type=int, default=10, help="Number of seeds per voxel to use for tracking.")

    main(parser.parse_args())