# halo-lidar

# Understanding of data

The received data has been processed by Ville

Data is in NetCDF format.

- some has only one value, they represent the settings, all the things that remain constant for the instrument

- some has lots of values, they are data collected by the instrument with respect to 'time' and 'range'

# Understanding of instrument

Instrument output is two variables, SNR and v. v_error is calculated from SNR and v

# Plot

Time array has the shape of (5328,)
Range array has the shape of (320,)
Data of each variable array has the shape of (320, 5328)
The pcolormesh function require the data array to be transposed. No clue why as it does not make sense. I think that this function has a long history, and matplotlib just try to respect it
