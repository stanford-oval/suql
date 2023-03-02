class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_chatbot(s: str):
    print(bcolors.OKGREEN + bcolors.BOLD + s + bcolors.ENDC)


def input_user() -> str:
    user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + 'User: ')
    while (not user_utterance.strip()):
        user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + 'User: ')
    print(bcolors.ENDC)
    return user_utterance
