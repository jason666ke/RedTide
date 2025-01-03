#2N_17.75N_105E_120.75E


import pygrib as pg
import numpy as np
from tqdm import tqdm
fimaly_fold = 'H:\wind_2019-2023'
target_fold = 'H:\paper3_dataset/2N_17.75N_105.25E_120.75E'
data_names = ['2019_10m_u_component_of_wind','2020_10m_u_component_of_wind',
			  '2021_10m_u_component_of_wind','2022_10m_u_component_of_wind',
			  '2023_10m_u_component_of_wind','2019_10m_v_component_of_wind',
			  '2020_10m_v_component_of_wind','2021_10m_v_component_of_wind',
			  '2022_10m_v_component_of_wind','2023_10m_v_component_of_wind']

print('begin')
for data_name in data_names:
	grbs = pg.open(fimaly_fold + '/' + data_name + '.grib')
	data = []
	for grb in tqdm(grbs):

		sub, lats, lons = grb.data(lat1=2,lat2=17.75,lon1=105,lon2=120.75)

		data.append(sub)

	data_np = np.stack(data,axis=0)
	np.save(target_fold+'/'+data_name+'.npy', data_np)
	print('finish_' + data_name)

print('finish_all')
