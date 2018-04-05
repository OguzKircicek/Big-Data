import json
from pprint import pprint

from geopy.geocoders import Nominatim

with open('C:/Users/oguzk/Desktop/2.json') as json_data:
    d = json.load(json_data)
    json_data.close()
dosya=open("C:/Users/oguzk/Desktop/1.2.csv","w")
counter=0
for i in d:

    geolocator = Nominatim(timeout=1500)
    counter=counter+1
    location = geolocator.geocode(i["name"])
    dosya.write(str(counter))
    dosya.write(location.address)
    dosya.write(json.dumps((location.latitude,location.longitude)))
    dosya.write("\n")
    print(location.address)
    print(location.latitude,location.longitude)


dosya.close()