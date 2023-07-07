from pymongo import MongoClient, ASCENDING
from prompt_continuation import llm_generate
import math

# set up the MongoDB connection
client = MongoClient('localhost', 27017)
db = client['yelpbot']
collection = db['yelp_data']
new_collection = db['yelp_data_four_locs']


class LocationLike:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
SF_LOCATION        = LocationLike(-122.419906, 37.7790262)
CUPERTINO_LOCATION = LocationLike(-122.0322895, 37.3228934)
SUNNYVALE_LOCATION = LocationLike(-122.036349 , 37.3688301)
PALOALTO_LOCATION  = LocationLike(-122.1598465, 37.4443293)


def distance(a : LocationLike, b : LocationLike):
    # print(a.__dict__)    
    R = 6371000
    lat1 = a.y
    lat2 = b.y
    lon1 = a.x
    lon2 = b.x
    
    def toRadians(deg):
        return deg * math.pi / 180.0

    φ1 = toRadians(lat1)
    φ2 = toRadians(lat2)
    Δφ = toRadians(lat2-lat1)
    Δλ = toRadians(lon2-lon1)

    x = math.sin(Δφ/2) * math.sin(Δφ/2) + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2) * math.sin(Δλ/2)
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))

    return R * c

    
def locationEquals(a : LocationLike, b : LocationLike) -> bool:
    if a.x == b.x and a.y == b.y:
        return True
    
    d = distance(a, b)
    return d <= 2500

if __name__ == "__main__":

    res = 0
    for location in [PALOALTO_LOCATION, CUPERTINO_LOCATION, SUNNYVALE_LOCATION, SF_LOCATION]:
        for i in collection.find():
            if "coordinates" not in i:
                continue
            if "latitude" not in i["coordinates"] or "longitude" not in i["coordinates"]:
                continue
            if None in [i["coordinates"]["longitude"], i["coordinates"]["latitude"]]:
                continue
            
            i_location = LocationLike(i["coordinates"]["longitude"], i["coordinates"]["latitude"])
        
            if locationEquals(i_location, location):
                res += 1
                print(i["name"])
                
                new_collection.insert_one(i)
                
            
    print(res)