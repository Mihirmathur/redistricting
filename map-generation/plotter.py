import geopandas as gpd
import pathlib
import urllib.request

states_filename = "tl_2017_us_state.zip"
states_url = "https://www2.census.gov/geo/tiger/TIGER2017/STATE/{}".format(
    states_filename)
states_file = pathlib.Path(states_filename)

zipcode_filename = "tl_2017_us_zcta510.zip"
zipcode_url = "https://www2.census.gov/geo/tiger/TIGER2017/ZCTA5/{}".format(
    zipcode_filename)
zipcode_file = pathlib.Path(zipcode_filename)

for data_file, url in zip([states_file, zipcode_file], [states_url, zipcode_url]):
    if not data_file.is_file():
        with urllib.request.urlopen(url) as resp, \
                open(data_file, "wb") as f:
            f.write(resp.read())

zipcode_gdf = gpd.read_file(f"zip://{zipcode_file}")
states_gdf = gpd.read_file(f"zip://{states_file}")

print(zipcode_gdf.head())
zipcode_gdf.plot()
