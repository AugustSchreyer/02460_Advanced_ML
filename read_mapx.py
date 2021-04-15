import h5py
import numpy as np
import logging
import pickle
import os
import matplotlib.pyplot as plt

class RamanExperimentData:
    """
    Class to hold Raman maps saved in mapx files - a flavour of HDF5 files.
    """
    def __init__(self, filenames):
        """
        Initialize class. Automatically loads mapx files in filenames (ITERABLE!) and saves size, wavenumbers and spectra
        """
        if len(filenames) > 0:
            self.maps = {f : [None, None, None] for f in filenames}

            for f in filenames:
                # spectrum, wavenumbers, mapsize
                self.maps[f][0], self.maps[f][1], self.maps[f][2] = self.read_mapx(f)

    def __getitem__(self, item):
        return self.maps[item]
    def read_mapx(self, filename):
        # read HDF5 file
        file = h5py.File(filename, 'r')

        # more than one acquired region.. sigh
        if len(file['Regions']) > 1:
            logging.warning(f"File {filename} has more than one region, only largest acquired region is imported")

        # find largest map and extract spectra
        N = 0
        info = file['Regions']
        for group in info.keys():
            dataset = info[group]['Dataset']
            N_cur = np.prod(dataset.shape[:2])
            if N_cur > N:
                # data should only be actually read when the dataset is sliced - otherwise only meta data regarding the map is read
                spectra = dataset[:]
                N = N_cur
        # get mapsize
        mapsize = (spectra.shape[1], spectra.shape[0])

        # extract wavenumbers
        metadata = file['FileInfo'].attrs
        wl_start = metadata['SpectralRangeStart']
        wl_end = metadata['SpectralRangeEnd']
        w = np.linspace(wl_start, wl_end, spectra.shape[2])

        # reshape spectra into regular size (N x W) - super weird, but it works!
        spectra = spectra.T.reshape((spectra.shape[2], np.prod(spectra.shape[:2])), order='F').T

        # close file
        file.close()

        return spectra, w, mapsize

    def save_pickle(self, filename):
        ans = 'y'
        if os.path.isfile(filename):
            logging.warning(f"File \"{filename}\" already exists. All data in the file will be overwritten")
            ans = input(f"Overwrite contents in {filename}? [y/n]")

        if ans == 'y':
            with open(filename, 'wb') as f:
                pickle.dump(self.maps, f)
            print(f"Finished saving in {filename}")
        else:
            print("Aborting...")

    def load_pickled(self, filename):
        with open(filename, 'rb') as f:
            self.maps = pickle.load(f)


if __name__ == '__main__':

    filenames = [
        "MAPS/0.025ppm_1.mapx",
    ]

    dat_class = RamanExperimentData(filenames=filenames)
    spectra, w, mapsize = dat_class.maps[filenames[0]]
    #spectra, w, mapsize = dat_class.read_mapx("1ppm_1.mapx")

    print(spectra.shape)
    print(w.shape)
    print(mapsize)

    for i in np.arange(0, spectra.shape[0], 500):
        plt.plot(w, spectra[i, 0:])
    
    spectra = spectra.reshape(*mapsize, -1)
    plt.plot(w, spectra[-1, 64, :])
    plt.show()

    print(spectra.shape)

    idx = np.argmin(np.abs(w - 270))

    plt.matshow(spectra[:, :, idx])
    plt.colorbar()
    plt.show()

    plt.plot(w[0:500], spectra[-1, 64, :500])

    plt.plot(spectra[-1, 64, :500])
    plt.show()


    






