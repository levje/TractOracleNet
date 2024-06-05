import subprocess
import argparse
from pathlib import Path
from score_tractogram import score_tractograms
from tempfile import TemporaryDirectory

fibercup_handle = "fibercup"
ismrm2015_handle = "ismrm2015"

def validate_input(dataset, tractogram_extension, scored_extension):
    if dataset not in [fibercup_handle, ismrm2015_handle]:
        raise ValueError(f"Invalid dataset. Please choose either '{fibercup_handle}' or '{ismrm2015_handle}'.")
    if tractogram_extension not in ["trk", "tck"]:
        raise ValueError("Invalid extension. Please choose either 'trk' or 'tck' for the tractogram extension.")
    if scored_extension not in ["trk", "tck"]:
        raise ValueError("Invalid extension. Please choose either 'trk' or 'tck'.")

def scil_tractogram_number_streamlines(tractogram_filepath):
    subprocess.run([
        "scil_tractogram_count_streamlines.py",
        str(tractogram_filepath)
    ])

def scil_resample_tractogram(tractogram_filepath, out_filepath, reference, nb_streamlines):
    subprocess.run([
        "scil_tractogram_resample.py",
        str(tractogram_filepath),
        str(nb_streamlines),
        str(out_filepath),
        "--reference", reference
    ])


def scil_tracking_local(algo, dataset, out_filepath, npv, step):
    subprocess.run([
        "scil_tracking_local.py",
        "--algo={}".format(algo),
        "--npv={}".format(npv),
        "--step={}".format(step),
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "{}/masks/{}_mask_wm.nii.gz".format(dataset, dataset),
        str(out_filepath),
    ])


def scil_tracking_pft(algo, dataset, out_filepath, npv, step):
    subprocess.run([
        "scil_tracking_pft.py",
        "--algo={}".format(algo),
        "--npv={}".format(npv),
        "--step={}".format(step),
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "{}/maps/{}_map_include.nii.gz".format(dataset, dataset),
        "{}/maps/{}_map_exclude.nii.gz".format(dataset, dataset),
        str(out_filepath),
    ])

def mrtrix_ifod2(dataset, out_filepath, nb, step):
    subprocess.run([
        "tckgen",
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        str(out_filepath),
        "-algorithm", "IFOD2",
        "-select", str(nb),
        "-step", str(step),
        "-minlength", "20",
        "-maxlength", "200",
        "-seed_image", "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "-nthreads", "4"
    ])

def mrtrix_sd_stream(dataset, out_filepath, nb, step):
    subprocess.run([
        "tckgen",
        "{}/fodfs/{}_fodf.nii.gz".format(dataset, dataset),
        str(out_filepath),
        "-algorithm", "SD_Stream",
        "-select", str(nb),
        "-step", str(step),
        "-minlength", "20",
        "-maxlength", "200",
        "-seed_image", "{}/masks/{}_interface.nii.gz".format(dataset, dataset),
        "-nthreads", "4"
    ])

def main(args):
    dataset = args.dataset
    tractogram_extension = args.extension
    scored_extension = args.scored_extension
    select = args.select
    step = args.step
    out_dir_name = args.out_dir_name
    npv = 10

    reference = "{}/anat/{}_T1.nii.gz".format(dataset, dataset) if dataset != fibercup_handle else "{}/dti/{}_fa.nii.gz".format(dataset, dataset)

    validate_input(dataset, tractogram_extension, scored_extension)

    tractograms_directory = Path("{}/{}".format(dataset, out_dir_name))
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

        with TemporaryDirectory() as temp_dir:
            out_filename = "{}_local_{}.{}".format(dataset, algo, tractogram_extension)
            inter_filename = "{}_local_{}_inter.{}".format(dataset, algo, tractogram_extension)
            out_filepath = tractograms_directory / out_filename
            inter_filepath = Path(temp_dir) / inter_filename

            print("Tracking with local {}".format(algo))
            scil_tracking_local(algo, dataset, inter_filepath, npv=npv, step=step)

            scil_tractogram_number_streamlines(inter_filepath)
 
            print("Resampling tractogram to {} streamlines.".format(select))
            scil_resample_tractogram(inter_filepath, out_filepath, reference, select)

        tractograms_filepaths.append(out_filepath)

    # PFT tracking
    scil_tracking_pft_algos = ["det", "prob"]
    for algo in scil_tracking_pft_algos:
        with TemporaryDirectory() as temp_dir:
            out_filename = "{}_pft_{}.{}".format(dataset, algo, tractogram_extension)
            inter_filename = "{}_pft_{}_inter.{}".format(dataset, algo, tractogram_extension)
            out_filepath = tractograms_directory / out_filename
            inter_filepath = Path(temp_dir) / inter_filename

            print("Tracking with pft {}".format(algo))
            scil_tracking_pft(algo, dataset, inter_filepath, npv=npv, step=step)

            scil_tractogram_number_streamlines(inter_filepath)

            print("Resampling tractogram to {} streamlines.".format(select))
            scil_resample_tractogram(inter_filepath, out_filepath, reference, select)

            tractograms_filepaths.append(out_filepath)

    # MRtrix IFOD2
    ifod2_out_filepath = tractograms_directory / "{}_ifod2.{}".format(dataset, tractogram_extension)
    print("Tracking with MRtrix IFOD2")
    mrtrix_ifod2(dataset, ifod2_out_filepath, nb=select, step=step)
    tractograms_filepaths.append(ifod2_out_filepath)

    # MRtrix SD_Stream
    sd_stream_out_filepath = tractograms_directory / "{}_sd_stream.{}".format(dataset, tractogram_extension)
    print("Tracking with MRtrix SD_Stream")
    mrtrix_sd_stream(dataset, sd_stream_out_filepath, nb=select, step=step)
    tractograms_filepaths.append(sd_stream_out_filepath)

    # Make sure the scoring directory exists
    scored_tractograms_directory = tractograms_directory / "scored_tractograms"
    if not scored_tractograms_directory.exists():
        scored_tractograms_directory.mkdir()

    # Score tractograms using tractometer, then fusing the scores of the filtered/unfiltered streamlines into a single tractogram.
    score_tractograms(
        reference,                                                  # reference image
        tractograms_filepaths,                                      # tractograms to score
        "{}/scoring_data/".format(dataset), # gt_config
        "{}/{}/scored_tractograms/".format(dataset, out_dir_name),       # out_dir
        scored_extension=scored_extension,
        is_ismrm=dataset==ismrm2015_handle
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and score tractograms for a given dataset.")
    parser.add_argument("dataset", type=str, help="Dataset to use for generating and scoring tractograms.", choices=[fibercup_handle, ismrm2015_handle])
    parser.add_argument("--extension", type=str, default="tck", help="Extension to use for the generated tractograms.", choices=["trk", "tck"])
    parser.add_argument("--scored_extension", type=str, default="trk", help="Extension to use for the scored tractograms.", choices=["trk", "tck"])
    parser.add_argument("--select", type=int, default=100000, help="Number of streamlines to select after filtering criterions are applied.")
    parser.add_argument("--step", type=float, default=0.75, help="Step size to use for tracking.")
    parser.add_argument("--out_dir_name", type=str, default="tractograms", help="Output directory name under <dataset>/<out_dir_name> for the generated tractograms.")
    main(parser.parse_args())
