import pygrib

file_path = "/mnt/g/sst_2015-2023/2017_sea_surface_temperature.grib"
grbs = pygrib.open(file_path)

for grb in grbs:
    print(grb)

# temp = grbs.select(name='2 metre temperature')[0]
# print("Variable details:", temp)

# data, lats, lons = temp.data()
# print("Data shape:", data.shape)
# print("Latitude range:", lats.min(), lats.max())
# print("Longitude range:", lons.min(), lons.max())

grbs.close()