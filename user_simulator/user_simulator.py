import argparse
from datetime import datetime
import logging
import os
import sys
import random
from typing import List
from pyGenieScript import geniescript as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from yelp_loop import (
    compute_next_turn,
    DialogueTurn,
    print_chatbot,
    dialogue_history_to_text,
)
from prompt_continuation import llm_generate

logger = logging.getLogger(__name__)

user_characters = [
    'You live in Mountain View, CA. Your high-school friend is visiting you from out of town, and has a 5-year old with him.',
    "You are a middle-aged women living in NYC.",
]


def simulate_one_user_turn(
    dialog_history: List[DialogueTurn], user_character: str
) -> str:
    return llm_generate(
        template_file="prompts/user.prompt",
        prompt_parameter_values={
            "dlg": dialog_history,
            "user_character": user_character,
        },
        engine="gpt-35-turbo",
        max_tokens=60,
        temperature=0.8,
        stop_tokens=["\n"],
        top_p=0.8,
        frequency_penalty=0.4,
        presence_penalty=0,
        postprocess=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--greeting",
        type=str,
        default="Hi! How can I help you?",
        help="The first thing the agent says to the user",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to write the outputs."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="text-davinci-003",
        choices=[
            "text-ada-001",
            "text-babbage-001",
            "text-curie-001",
            "text-davinci-002",
            "text-davinci-003",
            "gpt-35-turbo",
        ],
        help="The GPT-3 engine to use.",
    )  # choices are from the smallest to the largest model
    parser.add_argument(
        "--num_turns",
        type=int,
        required=True,
        help="Number of turns to simulate per dialog.",
    )
    parser.add_argument(
        "--num_dialogs", type=int, required=True, help="Number of dialogs to simulate."
    )
    parser.add_argument(
        "--no_logging",
        action="store_true",
        help="Do not output extra information about the intermediate steps.",
    )

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(
            level=logging.CRITICAL, format=" %(name)s : %(levelname)-8s : %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
        )

    # the agent starts the dialogue
    genie = gs.Genie()
    dlgHistory = [DialogueTurn(agent_utterance=args.greeting)]
    genieDS, genie_aux = "null", []

    print_chatbot(dialogue_history_to_text(dlgHistory, they="User", you="Chatbot"))

    try:
        genie.initialize("localhost", "yelp")

        with open(args.output_file, "a") as output_file:
            for dlg_id in range(args.num_dialogs):
                for turn_id in range(args.num_turns):
                    # sample a user characteristic
                    user_utterance = simulate_one_user_turn(
                        dlgHistory, random.choice(user_characters)
                    )

                    # this is single-user, so feeding in genieDS and genie_aux is unnecessary, but we do it to be consistent with backend_connection.py
                    dlgHistory, response, gds, gaux, _ = compute_next_turn(
                        dlgHistory,
                        user_utterance,
                        genie,
                        genieDS=genieDS,
                        genie_aux=genie_aux,
                        engine=args.engine,
                    )
                    if genieDS != "null":
                        # update the genie state only when it is called. This means that if genie is not called in one turn, in the next turn we still provide genie with its state from two turns ago
                        genieDS = gds
                        genie_aux = gaux

                # write the dialog to the output file
                output_file.write(
                    "=====\n"
                    + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    + "\n"
                    + dialogue_history_to_text(dlgHistory, they="User", you="Chatbot")
                    + "\n"
                )

    finally:
        # not necessary to close genie, but is good practice
        genie.quit()
