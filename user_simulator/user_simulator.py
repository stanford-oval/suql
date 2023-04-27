import argparse
from datetime import datetime
import logging
import os
import string
import sys
import random
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from yelp_loop import (
    DialogueTurn,
    dialogue_history_to_text,
)
from prompt_continuation import llm_generate
from backend_connection import BackendConnection


logger = logging.getLogger(__name__)

user_characters = [
    "You live in Mountain View, CA. Your high-school friend is visiting you from out of town, and has a 5-year old with him.",
    "You are a middle-aged women living in NYC.",
    "You are always looking for the trendy restuarants in town.",
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

    connection = BackendConnection()

    with open(args.output_file, "a") as output_file:
        for dlg_idx in range(args.num_dialogs):
            dialog_id = (
                "simulated_"
                + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                + "_"
                + str(dlg_idx)
            )

            dlgHistory = []
            for turn_id in range(args.num_turns):
                # sample a user characteristic
                user_utterance = simulate_one_user_turn(
                    dlgHistory,
                    random.choice(user_characters),
                )

                _, dlgItem, _ = connection.compute_next(
                    dialog_id,
                    user_utterance,
                    turn_id,
                )
                print("dlgItem = ", dlgItem.to_text())
                dlgHistory.append(dlgItem)

            # write the dialog to the output file
            output_file.write(
                "=====\n"
                + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                + " dialog_id="
                + dialog_id
                + "\n"
                + dialogue_history_to_text(dlgHistory, they="User(sim)", you="Chatbot")
                + "\n"
            )
