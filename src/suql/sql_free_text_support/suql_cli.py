import argparse
import json
import os
import atexit
from suql.sql_free_text_support.execute_free_text_sql import suql_execute
import pkg_resources

def setup_readline(history_file: str | None = None) -> None:
    """
    Enable readline-powered line editing + history for `input()` when running in a TTY.
    On Linux/macOS this provides left/right cursor movement and up/down history.
    """
    try:
        import readline  # noqa: F401
    except Exception:
        # e.g. Windows or Python built without readline support
        return

    if history_file is None:
        history_file = os.path.expanduser("~/.suql_history")

    try:
        import readline

        readline.set_history_length(1000)
        if os.path.exists(history_file):
            readline.read_history_file(history_file)
        atexit.register(readline.write_history_file, history_file)
    except Exception:
        # Never fail the CLI because of readline/history issues
        return

def get_version():
    try:
        return pkg_resources.get_distribution("suql").version
    except pkg_resources.DistributionNotFound:
        return "unknown"

def execute_query(sql_query, table_mappings, database, embedding_server_address, source_file_mapping, disable_try_catch, disable_try_catch_all_sql, fts_fields, llm_model_name, max_verify, loggings, log_filename, select_username, select_userpswd, create_username, create_userpswd):
    results, _, _ = suql_execute(
        sql_query,
        table_mappings,
        database,
        fts_fields=fts_fields,
        llm_model_name=llm_model_name,
        max_verify=max_verify,
        loggings=loggings,
        log_filename=log_filename,
        disable_try_catch=disable_try_catch,
        disable_try_catch_all_sql=disable_try_catch_all_sql,
        embedding_server_address=embedding_server_address,
        select_username=select_username,
        select_userpswd=select_userpswd,
        create_username=create_username,
        create_userpswd=create_userpswd,
        source_file_mapping=source_file_mapping,
    )
    print(results)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Interactive shell for executing suql queries.")
    
    # Arguments to initialize the shell
    parser.add_argument(
        "--table-mappings", 
        type=json.loads, 
        required=True, 
        help="JSON string representing table mappings (e.g., '{\"table1\": \"id1\", \"table2\": \"id2\"}')"
    )
    parser.add_argument(
        "--database", 
        type=str, 
        required=True, 
        help="Database name or identifier"
    )
    parser.add_argument(
        "--embedding-server-address", 
        type=str, 
        required=False, 
        default="http://127.0.0.1:8501", 
        help="Embedding server address (default: http://127.0.0.1:8501)"
    )
    parser.add_argument(
        "--disable-try-catch", 
        action="store_true", 
        help="Disable try-catch for the execution (default: False)"
    )
    parser.add_argument(
        "--disable-try-catch-all-sql", 
        action="store_true", 
        help="Disable try-catch for all SQL (default: False)"
    )
    parser.add_argument(
        "--source-file-mapping", 
        type=json.loads, 
        required=False, 
        default={}, 
        help="JSON string representing source file mappings (default: {})"
    )
    parser.add_argument(
        "--fts-fields", 
        type=json.loads, 
        required=False, 
        default=[], 
        help="List of full-text search fields (default: [])"
    )
    parser.add_argument(
        "--llm-model-name", 
        type=str, 
        required=False, 
        default="gpt-3.5-turbo-0125", 
        help="Language model name (default: gpt-3.5-turbo-0125)"
    )
    parser.add_argument(
        "--max-verify", 
        type=int, 
        required=False, 
        default=20, 
        help="Maximum number of verifications (default: 20)"
    )
    parser.add_argument(
        "--loggings", 
        type=str, 
        required=False, 
        default="", 
        help="Logging options (default: '')"
    )
    parser.add_argument(
        "--log-filename", 
        type=str, 
        required=False, 
        default=None, 
        help="Log file name (default: None)"
    )
    parser.add_argument(
        "--select-username", 
        type=str, 
        required=False, 
        default="select_user", 
        help="Username for select operations (default: select_user)"
    )
    parser.add_argument(
        "--select-userpswd", 
        type=str, 
        required=False, 
        default="select_user", 
        help="Password for select operations (default: select_user)"
    )
    parser.add_argument(
        "--create-username", 
        type=str, 
        required=False, 
        default="creator_role", 
        help="Username for create operations (default: creator_role)"
    )
    parser.add_argument(
        "--create-userpswd", 
        type=str, 
        required=False, 
        default="creator_role", 
        help="Password for create operations (default: creator_role)"
    )

    args = parser.parse_args()

    setup_readline()
    
    # Welcome message
    print(f"SUQL REPL loop, pip version {get_version()}")
    print("Type your SUQL queries to execute. Type 'exit' to quit.")
    
    # Start the REPL loop
    while True:
        try:
            sql_query = input("suql> ")
            if sql_query.lower() in {"exit", "quit"}:
                print("Exiting the SUQL Shell. Goodbye!")
                break
            
            # Execute the query
            execute_query(
                sql_query,
                args.table_mappings,
                args.database,
                args.embedding_server_address,
                args.source_file_mapping,
                args.disable_try_catch,
                args.disable_try_catch_all_sql,
                args.fts_fields,
                args.llm_model_name,
                args.max_verify,
                args.loggings,
                args.log_filename,
                args.select_username,
                args.select_userpswd,
                args.create_username,
                args.create_userpswd,
            )
        except KeyboardInterrupt:
            print("\nExiting the SUQL Shell. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()