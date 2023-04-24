include ./API_KEYS

### Input variables
engine ?= text-davinci-003
num_simulation_turns ?= 5
num_simulation_dialogs ?= 1

.PHONY: yelpbot genie-server start-backend simulate-user

yelpbot: yelp_loop.py
	python yelp_loop.py \
	--engine $(engine) \
	--output_file yelpbot.log

genie-server: run_genie_server.py
	python run_genie_server.py

start-backend: backend_connection.py
	python backend_connection.py \
	--engine $(engine)

simulate-user:
	python user_simulator/user_simulator.py \
	--engine $(engine) \
	--output_file user_simulator/simulated_dialogs.txt \
	--num_turns $(num_simulation_turns) \
	--num_dialogs $(num_simulation_dialogs)