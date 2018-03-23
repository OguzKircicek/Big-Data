rating=LOAD '/user/pigdata/u.data' AS (userID:int,movieID:int,rating:int,ratingTime:int);

metadata=LOAD '/user/pigdata/u.item' USING PigStorage('|') AS (movieID:int,movieTitle:chararray,releaseDate:chararray,videoRelease:chararray,imdbLink:chararray);

nameLookup=FOREACH metadata generate movieID,movieTitle,ToUnixTime(ToDate(releaseDate,'dd-MMM-yyyy')) AS releaseTime;

ratingsByMovie=GROUP ratings By movieID; 

avgRatings=FOREACH ratingsByMovie GENERATE group AS movieID,AVG(ratings.rating) AS avgRating;

fiveStarMovies=FILTER avgRatings BY avgRating>4.0;

fiveStarWithData=JOIN fiveStarMovies BY movieID,nameLookup BY movieID;

oldestFiveStarMovies=ORDER fiveStarWithData BY nameLookup::releaseTime;

DUMP oldestFiveStarMovies;