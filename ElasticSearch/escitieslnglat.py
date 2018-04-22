from geopy import geocoders
import json
from pprint import pprint
import io
import time
import pymysql.cursors
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from elasticsearch import Elasticsearch

db=pymysql.connect("localhost","root","","proje",use_unicode=True,charset="utf8")
cursor=db.cursor()

g = geocoders.GoogleV3(api_key='*********************')

es=Elasticsearch([{'host':'localhost','port':9200}])

#do the geocode


#some things you can get from the result


sorgu="SELECT name FROM cities"
cursor.execute(sorgu)

db.commit()
data=cursor.fetchall()

dosya=open("C:/Users/oguzk/Desktop/123.csv","a+",newline='',encoding="utf-8")

i=0
counter=0

for row in data:

    if row[0]!=None:


        location = g.geocode(row[0],timeout=10)
        if location !=None:
            print(location.address)

            print(location.latitude, location.longitude)
            e1 = {
                "cities": row[0],
                "address": location.address,
                "latlng": (location.latitude, location.longitude)
            ,

            }
            counter=counter+1
            res = es.index(index='proje', doc_type='world', id=counter, body=e1)

        else:
            print("yok")





       # print(row[1])



dosya.close()
