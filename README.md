# halo-lidar

# Understanding of data

The received data has been processed by Ville

Data is in NetCDF format.

- some has only one value, they represent the settings, all the things that remain constant for the instrument

- some has lots of values, they are data collected by the instrument with respect to 'time' and 'range'

# Understanding of instrument

Instrument output is two variables, SNR and v. v_error is calculated from SNR and v
