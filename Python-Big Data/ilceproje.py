import json
from pprint import pprint
import io
import time

from geopy.geocoders import Nominatim

with open('C:/Users/oguzk/Desktop/2.json') as json_data:
    d = json.load(json_data)
    json_data.close()
dosya=open("C:/Users/oguzk/Desktop/123.csv","a+",newline='',encoding="utf-8")
counter=82

for i in d:
    for a in i["counties"]:
        geolocator = Nominatim(timeout=1500)


        location = geolocator.geocode(a)

        dosya.write(str(counter))
        dosya.write(location.address)
        dosya.write(json.dumps((location.latitude,location.longitude)))
        dosya.write("\n")
        counter = counter + 1
        print(location.address)
        print(location.latitude,location.longitude)


dosya.close()
