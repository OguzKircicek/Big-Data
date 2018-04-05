Merhaba, 

Türkiyenin il ve ilçelerini kapsayan bir projedir

Ýl ve ilçe verileri json verisinde parse edilerek Python dilinde ki geopy librarysini kullanarak,
latitude ve longitude deðerleri bulunmuþtur.

Deðerlere id deðeri atandý ve csv dosyasýna her deðer kayýt edildi. Source codelar dosyanýn içinde olucak. 

Hadoop sistemine atmaya gelince Ben zaten VM'de (Virtual Machine) Cloudera kullanýcýsýyým onun için Hadoop-HDFS dosya sistemine,

direkt eriþebiliyorum.Cloudera sistemi üzerinde terminalde hdfs dfs -copyFromLocal /home/cloudera/Desktop/proje.csv /user/proje  þeklinde-

projemi hdfs sistemine attým.

