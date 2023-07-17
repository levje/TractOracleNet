import h5py
import numpy as np

from nibabel.streamlines import Tractogram
from torch.utils.data import Dataset


class StreamlineDataset(Dataset):
    """
    TODO
    """

    def __init__(
        self,
        file_path: str,
        noise: float = 0.1,
        flip_p: float = 0.5,
        dense: bool = False,
        device=None
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.noise = noise
        self.flip_p = flip_p
        self.dense = dense
        self.n_f = 0
        with h5py.File(self.file_path, 'r') as f:
            self.subject_list = list(f.keys())
            self.indexes = \
                self._build_indexes(f)
            self.input_size = self._compute_input_size()

    def _build_indexes(self, dataset_file):
        """ TODO
        """
        print('Building indexes')
        set_list = list()

        f = self.archives
        print(list(f.keys()))
        streamlines = f['streamlines']['data']
        for i in range(len(streamlines)):
            k = i

            set_list.append(k)

        return set_list

    def _compute_input_size(self):
        batch = self._get_one_input()
        L, P = batch[0].shape
        return L * P

    @property
    def archives(self):
        """ TODO
        """
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()
            print('Destructor called, File closed.')

    def _get_one_input(self):
        """ TODO
        """

        state_0, *_ = self[[0, 1]]
        self.f.close()
        del self.f
        return state_0

    def __getitem__(self, indices):
        """ TODO

        :arg
            indices: TODO
        :return
            A tuple (input, target)
        """

        f = self.archives

        hdf_subject = f['streamlines']
        data = hdf_subject['data']
        scores_data = hdf_subject['scores']

        start, end = indices[0], indices[-1] + 1

        # Handle rollover indices
        if start > end:
            batch_end = max(indices)
            batch_start = min(indices)
            streamlines = np.concatenate(
                (data[start:batch_end], data[batch_start:end]), axis=0)
            score = np.concatenate(
                (scores_data[start:batch_end], scores_data[batch_start:end]),
                axis=0)
        # Slice as usual
        else:
            streamlines = hdf_subject['data'][start:end]
            score = hdf_subject['scores'][start:end]

        # Flip streamline for robustness
        if np.random.random() < self.flip_p:
            streamlines = np.flip(streamlines, axis=1).copy()

        # Add noise to streamline points for robustness
        if self.noise > 0.0:
            dtype = streamlines.dtype
            streamlines = streamlines + np.random.normal(
                loc=0.0, scale=self.noise, size=streamlines.shape
            ).astype(dtype)

        # Convert the streamline points to directions
        # Works really well
        dirs = np.diff(streamlines, axis=1)

        return dirs, score

    def __len__(self):
        """
        Return the length of the dataset, i.e. the number
        of streamlines in the dataset.
        """
        return int(len(self.indexes))

    def render(
        self,
        streamline
    ):
        """ Debug function

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        tractogram = Tractogram(
            streamlines=streamline,
            data_per_streamline={
                'seeds': streamline[:, 0, :]
            })

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines)
        scene.add(stream_actor)
        scene.reset_camera_tight(0.95)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
