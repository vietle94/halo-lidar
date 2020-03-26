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

# Briefing from Ville

Should take a box where no signal of random noise and use that box to determine SNR limit. Then use SNR limit 2/3 times bigger.
Depends on the averaging time – changes with instruments so need to redo with each instrument.
Check this once a year if instrument doesn’t change.


Make a code
-	One function that plots everything for one day, then click enter and plots the next day
-	Need to be able to choose interesting time height range in the data – click mouse on plot and know coordiantes of rectangle – and retrieve the data from the data file
-	Calc SD of SNR in that range – get SNR range. SNR is given as 1+SNR (need to -1 get get plain SNR). Get SD and then decide if want to use 2 sigma or 3 sigma limit.
-	When interesting case try both 2 and 3 and see what works better
-	Ville normally uses 2 sigma but need to try and see what it looks like

Depol in liquid cloud
-	Plot data and find a nice cloud – choose the cloud on the plot and see what the profiles look like – find depol values at maximum SNR in cloud base
-	Plot a histogram of that
-	Pick a part of the cloud that stays at about the same height
-	Do multiple – for period of cloud to get statitstics – will be noisy
-	Plot histogram – the mean should be around 1%
-	Want mean and SD of cloud
-	For lots of clouds throught year – want to see if instruemtns drift or remain and if any outliers
-	Likely to be  adifference between XR and non – non should be closer to 1%


Liquid cloud has high beta 10^4 roughly. Of thick cloud will fully attenuate in 200m or less. Will only see cloud base.
For ice cloud not as thick optically so will see distance into the ice cloud – hundreds of m. beta will be lower than for a liquid cloud and depol value much higher as non spherical.

For XR for liquid clouds better to pick something at 2-3 km


For what each of the things are – need to find attriubute table in net cdf
Data fields:
Raw = not applied snr
Beta raw – attenuated backscatter corrected for telescope range etc – optical thickness
V raw – radial velocity
V error – caclculated uncertainity in radial velocity calculated from SNR (shoudlnt need)
Co signal – SNR +1
Cross  signal – cross polar snr +1
Depo raw – cross/co snr
Averaged = 10 times longer than other  values ( a few mins)


If have good aerosol cases should average for longer time periods. Figure out what the best averging time is for each case


In matlab the function is called g input – find equivalent in python.
-	Input two point to get corners of box


# next
remove first three columns (each has 5320 data points) for all data
negative depo values are noise
FIND PURE LIQUID CLOUDS
go through each time stamp, and plot the depo values vs range. Same time stamp for the snr
get max snr in the plot, find its corresponding depo
take the mean and sd of depo values


find how much co-signal leak to cross?? not related

choose liquid cloud only

beta has some artifact, changes with temperature.

devide num pulses by 15000 to get numpulses/second
