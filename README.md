# PoC
Probability of Collision computation tool exploiting Vallado's method.
This tool computes the PoC between two object each defined by a vector of initial Keplerian elements and main characteristics.
To run the tool some python packages and additional files are needed: in your therminal run the command: ' pip install -r requirements.txt ' to install the main packages needed, if others are missing you can install them manually or add them to the list in the file. 
The other files needed are:
Spice ephemeris: de440.bsp
A file needed to convert UTC in ephemeris time (ET): naif0012.tls
Text Frame kernel for itfr93 frame definition: earth_assoc_itrf93.tf
Earth orientation file: earth_000101_240704_240410.bpc

## main.py
This python file contains all the config parameters that need to be changed to start a new analysis, fill in all the info and run the script.
