import numpy as np
from scipy import signal 
RELATIVE = 'relative'
ABSOLUTE = 'absolute'

def get_power_spectrum(data, orp_epoch_length, fs, freq_conf, logger = None):

    if logger:
        printf = logger
    else:
        printf = print
    
    step_fft = 0.33333333
    n_per_seg = orp_epoch_length * fs
    f, spectrum = signal.welch(data, 
                               fs=fs, 
                               window='hamming', 
                               nperseg=n_per_seg, 
                               noverlap=0, 
                               nfft=n_per_seg, 
                               scaling='spectrum', 
                               axis=-1, 
                               average='mean')

    assert np.isclose(f[1] - f[0], step_fft)
    assert spectrum.shape[0] == len(data)
    assert spectrum.shape[1] == len(f)
    
    power_type = freq_conf.power_type
    bands = freq_conf.sub_frequency_range
    if power_type == RELATIVE:
        total_power = np.sum(spectrum, axis=1)
        assert total_power.shape[0] == len(data)
        printf(f'power_type: {RELATIVE} => devided by total power from {f[0]}-{f[-1]} ({total_power})')

    results = {}
    for band in bands:
        band_name = list(band.keys())[0]
        low, high = list(band.values())[0]
          
        start_index = next(i for i in range(len(f)) if np.isclose(f[i], low) or f[i] >= low)
        end_index = next(i for i in range(len(f)-1, -1, -1) if np.isclose(f[i], high) or f[i] <= high) + 1

        selected_f = f[start_index:end_index]
        selected_p = spectrum[:, start_index:end_index]
        printf(band, selected_f[0], '-', selected_f[-1])
        assert np.isclose(selected_f[0], low) or selected_f[0] >= low
        assert np.isclose(selected_f[-1], high) or selected_f[-1] <= high

        sum_power = np.sum(selected_p, axis=-1)
        assert len(sum_power) == len(data)

        if power_type == RELATIVE:
            results[band_name] = sum_power / total_power
        elif power_type == ABSOLUTE:
            results[band_name] = sum_power
            
        assert len(results[band_name]) == len(data)
    
    return results
