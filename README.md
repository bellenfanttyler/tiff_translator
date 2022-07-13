# tiff_translator
This script takes in two .tif files (a source and a target) of the SAME geographical area (lat/long), and corrects the source .tif to match the feature locations of the target .tif

Installation:
Create a conda env using the environment.yml file. Activate this environment before calling the function.

Example invocation:
python tifftranslator_RevA.py -s SKYWATCH_SS_PS_20220326T1003_TC_Tile_0_0_oDhJ.tif -t cliptiffcorrect.tif
