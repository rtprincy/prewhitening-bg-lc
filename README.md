This is an iterative prewhitening tool for BlackGEM light curves. 

The following need to be manually specified before running the code. 

- Specify the light curve directory and filename as here: lc=fits.open("BG_lightcurves/lightcurves_4targets.fits")
- Then select the object ID. Here it is the Gaia DR3 ID (e.g. source_ids=[3600841623951744640])
- Finally, select the filter (e.g. passband=‘I’)

There are other parameters that might be useful to define, such as the frequency search range (f0 and fn) and the number of iterations (n_iterations = 10).

This code is still being improved so that it can be run without editing the code internally as shown above. 
Feel free to get in touch if you have any suggestions.


