from suql.free_text_fcns_server import start_free_text_fncs_server

host = "127.0.0.1"
port = 8500
start_free_text_fncs_server(host=host, port=port, engine="gpt-3.5-turbo-0125")
