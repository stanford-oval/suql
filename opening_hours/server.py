import flask, util
from flask import request, Flask
host = "127.0.0.1"
port = 8500
review_server_address = 'http://{}:{}'.format(host, port)
app = Flask(__name__)
chat = util.Chat()
import random
rn = random.randint
STRESS_TEST = False
truth = [ [ [ set() for minutes in range(60)] for hours in range(24)] for days in range(7)]
if STRESS_TEST == True:
    truth = [ [ [ set(x for x in range(16,8500)) for minutes in range(60)] for hours in range(24)] for days in range(7)]
    
RESTAURANTS = 15
# for each restaurant, let H be their segmented hours:

def index_to_segments(string_segment):
    #print(string_segment, "BOO")
    if not string_segment:
        return []
    string_segment = string_segment[0]
    split_string_segments = string_segment.split("-")
    segments_separated = [list(map(int,s.split("."))) for s in split_string_segments]
    segments = [[s[0], [[s[1],s[2]],[s[3],s[4]]]] for s in segments_separated] 
    #print(segments)
    return segments

indices = util.fetch_SQL("SELECT opening_hours_index FROM restaurants ORDER BY CAST(id AS INTEGER)")
print(indices)
for _ in range(1, RESTAURANTS+1):
    print("RESTAURANT ",_)
    restaurant_id = _
    
    if restaurant_id  % 100 ==0:
        print(restaurant_id) 
    
    H = index_to_segments(indices[_-1])
    print("INDEX", index_to_segments(indices[_-1]))
    for segment in H:
        day = segment[0]
        segment_start = segment[1][0]
        starting_hour = segment_start[0]
        starting_minute = segment_start[1]
        segment_end = segment[1][1]
        ending_hour = segment_end[0]
        ending_minute = segment_end[1]
        done = 0
        for hour in range(starting_hour, ending_hour + 1):
            if done: break
            for minute in range(0, 60):
                if hour == starting_hour and minute <= starting_minute:
                    pass
                elif hour == ending_hour and minute == ending_minute:
                    done = 1
                    break
                else: 
                    truth[day][hour][minute].add(restaurant_id) 
#for hour in truth[0]:
#    print(hour, "\n")
def search_segment(segment):
    #print(segment)
    segment = segment.split(".")
    segment = [int(x) for x in segment]
    #print(segment)
    day = segment[0]
    starting_hour = segment[1]
    starting_minute = segment[2]
    ending_hour = segment[3]
    ending_minute = segment[4]
    open_restaurants = set()
    #print(segment_start)
    #print(segment_end)
    #print(starting_hour, starting_minute)
    #print(ending_hour, ending_minute)
    done = 0
    for hour in range(starting_hour, min(ending_hour+1,24)):
        if done == 1:
            break
        for minute in range(0, 60):
            current_open_restaurants = truth[day][hour][minute]
            time = (day,hour,minute)
            #print(time, current_open_restaurants)
            if hour == starting_hour and minute < starting_minute:
                pass
            elif hour == ending_hour and minute == ending_minute:
                done = 1
                break
            else:
                open_restaurants = open_restaurants.union(current_open_restaurants) 
    open_restaurants = list(open_restaurants)
    print(open_restaurants)
    open_restaurants = [x for x in open_restaurants if x <=15]
    print(open_restaurants)
    return open_restaurants



def search_segments(segments):
    print(segments)
    for segment in segments:
        search_segment(segment)

@app.route("/opening_hours_search", methods=['POST'])
def opening_hours_search():	
    data = request.get_json()
    open_restaurants = search_segment(data["search_interval"])
    answer = open_restaurants 
    res = {"result": answer}
    return res
if __name__ == "__main__":
    app.run(host=host, port=port)
