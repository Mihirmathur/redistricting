# Code for Redistricting using Active Contours

Install [morphsnakes](https://github.com/pmneila/morphsnakes) using
```
pip install morphsnakes
```

To try out the example type:
```
cd model
python morphsnakesExample.py
```

# Converting dbf file to csv

- You will need Python2 for conversion.
- Place all .dbf files in map-generation/
- First cd into map-generation/dbfpy-2.3.1 and run:
```
sudo python setup.py install
```

- Conversion step:
```
python data_processing.py rg_match_p10_01.dbf 
```
# Downloads
- Pollvote stats by block: https://statewidedatabase.org/pub/data/D10/reg_stats/p10/statewide_pollvote_stats_by_block.dbf

# Plotting Resources/Ideas 
- https://n8henrie.com/uploads/2017/11/plotting-us-census-data-with-python-and-geopandas.html
- https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
