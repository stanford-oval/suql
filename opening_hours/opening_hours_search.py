import random
rn = random.randint
truth = [ [ [ set() for minutes in range(60)] for hours in range(24)] for days in range(7)]

RESTAURANTS = 1800
# for each restaurant, let H be their segmented hours:
for _ in range(RESTAURANTS):
    #print("RESTAURANT", _)
    restaurant_id = _
    if restaurant_id  % 100 ==0:
        print(restaurant_id) 
    H = [
            [i,[[0,0],[24,0]]] for i in range(7)
        ] 
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
                if hour == starting_hour and minute < starting_minute:
                    pass
                elif hour == ending_hour and minute == ending_minute:
                    done = 1
                    break
                else: 
                    truth[day][hour][minute].add(restaurant_id) 
#for hour in truth[0]:
#    print(hour, "\n")

segments = [
    [i, [[0,0], [23,59]]] for i in range(7)
]
print(segments)
for segment in segments:
    print(segment)
    day = segment[0]
    segement_start = segment[1][0]
    segment_end = segment[1][1]
    starting_hour = segment_start[0]
    starting_minute = segment_start[1]
    ending_hour = segment_end[0]
    ending_minute = segment_end[1]
    open_restaurants = set()
    print(segment_start)
    print(segment_end)
    print(starting_hour, starting_minute)
    print(ending_hour, ending_minute)
    done = 0
    for hour in range(starting_hour, ending_hour + 1):
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

    print(open_restaurants)


