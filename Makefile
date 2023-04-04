include ./API_KEYS

### Input variables
engine ?= text-davinci-003

.PHONY: yelpbot genie-server start-backend

### yelpbot
yelpbot: yelp_loop.py yelp_conversation.prompt
	python yelp_loop.py \
	--engine $(engine) \
	--output_file yelpbot.log

genie-server: run_genie_server.py
	python run_genie_server.py

start-backend: backend_connection.py
	python backend_connection.py \
	--engine $(engine)