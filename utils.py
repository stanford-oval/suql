import spacy
nlp = spacy.load('en_core_web_sm')
import hashlib
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

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


def chunk_text(text, k, use_spacy=True):
    """
    Chunk a string of text into a list of k-token sized strings

    :param text: string of all text to chunk
    :param k: int representing size of each chunk
    :return: a list of k-token-sized chunks of the original text
    """
    if use_spacy:
        if text == "":
            return [""]
        # in case of using spacy, k is the minimum number of words per chunk
        chunks = [i.text for i in nlp(text).sents]
        res = []
        carryover = ""
        for i in chunks:
            if len((carryover + i).split()) < k:
                # chunks stripped the spaces, so when we need to append
                # we append a space for preparation
                carryover = carryover + i + " "
            else:
                res.append(carryover + i)
                carryover = ""
        if carryover != "":
            res.append(carryover.rstrip())
        return res
            
    all_chunks = []
    counter = 0
    chunk = []

    split = text.split()
    
    for word in split:
        chunk.append(word)
        counter += 1

        if counter == 100:
            chunk = ' '.join(chunk)
            all_chunks.append(chunk)
            chunk = []
            counter = 0

    if chunk != []:
        chunk = ' '.join(chunk)
        all_chunks.append(chunk)

    return all_chunks


def linearize(document, k):
    """
    Takes in database information about a restaurant, and converts it into a linearized format as
    discussed in https://aclanthology.org/2022.findings-naacl.115.pdf
    The function also chunks it into k-token sized strings and returns it in a list
    If it is missing categories, it will return an empty list
    
    :param document: the JSON object of a restaurant's information
    :param k: int representing size of each chunk
    :return: a list of k-token-sized chunks (str) representing the linearized format of the restaurant
    """
    def convert_price(dollars):
        dollar_amt = ['', '$', '$$', '$$$', '$$$$']
        english = ['', 'cheap', 'moderate', 'expensive', 'luxury']

        for i, d in enumerate(dollar_amt):
            if dollars == d:
                return english[i]

    def convert_address(address):
        address_string = ""

        for i, line in enumerate(address):
            if i < len(address) - 1:
                address_string += (line + ', ')
            else: 
                address_string += line

        return address_string

    def convert_hours(hours):
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        hours_string = ""

        for i, day in enumerate(hours):
            actual_day = day_name[day['day']]
            start = day['start']
            end = day['end']
            if i < len(hours) - 1:
                hours_string += (actual_day + ' ' + start + ' ' + end + ', ')
            else:
                hours_string += (actual_day + ' ' + start + ' ' + end)
        return hours_string

    def convert_reviews(reviews):
        reviews_string = ""
        for i, review in enumerate(reviews):
            if i < len(reviews) - 1:
                reviews_string += (review + ', ')
            else:
                reviews_string += review
        return reviews_string
    

    name = document['name']
    cuisines = [c['title'] for c in document['categories']]
    price = convert_price(document['price'])
    rating = document['rating']
    num_reviews = document['review_count']
    address = convert_address(document['location']['display_address'])
    dishes = [dish[0] for dish in document['dishes']]
    phone_number = document['display_phone']
    opening_hours = convert_hours(document['hours'][0]['open']) if document['hours'] != "" else ""
    reviews = convert_reviews(document['reviews'])
    
    linearized = ""

    linearized += 'name, ' + name + '\n'

    linearized += 'cuisines, ' 
    for i, c in enumerate(cuisines):
        if i < len(cuisines) - 1:
            linearized += c + ', '
        else:
            linearized += c
    linearized += '\n'

    linearized += 'price, ' + price + '\n'

    linearized += 'rating, ' + str(rating) + '\n'

    linearized += 'num_reviews, ' + str(num_reviews) + '\n'

    linearized += 'address, ' + address + '\n'

    linearized += 'dishes, ' 
    for i, d in enumerate(dishes):
        if i < len(dishes) - 1:
            linearized += d + ', '
        else:
            linearized += d
    linearized += '\n'

    linearized += 'phone_number, ' + phone_number + '\n'

    linearized += 'opening_hours, ' + opening_hours + '\n'

    linearized += 'reviews, ' + reviews + '\n'

    linearized = chunk_text(linearized, k)

    return linearized
    
def compute_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

# this is a restaurants database specific function, which excludes a bunch of fields from
# the final response
def if_usable_restaurants(field : str):
    NOT_USABLE_FIELDS = [
        "reviews",
        "id",

        # schematized fields        
        "ambiance",
        "specials",
        "reservation_info",
        "nutrition_info",
        "signature_cocktails",
        "has_private_event_spaces",
        "promotions",
        "parking_options",
        "game_day_specials",
        "live_sports_events",
        "dress_code",
        "happy_hour_info",
        "highlights",
        "service",
        "has_outdoor_seating",
        "drinks",
        "dietary_restrictions",
        "experience",
        "nutritious_options",
        "creative_menu",
        "has_student_discount",
        "has_senior_discount",
        "local_cuisine",
        "trendy",
        "wheelchair_accessible",
        "noise_level",
        "kids_menu",
        "childrens_activities",
        "if_family_friendly",
        "wait_time",
        "has_live_music",
        "serves_alcohol",
        "michelin",
        "accomodates_large_groups",
        
        # the citation fields        
        "ambiance_citation",
        "specials_citation",
        "reservation_info_citation",
        "nutrition_info_citation",
        "signature_cocktails_citation",
        "has_private_event_spaces_citation",
        "promotions_citation",
        "parking_options_citation",
        "game_day_specials_citation",
        "live_sports_events_citation",
        "dress_code_citation",
        "happy_hour_info_citation",
        "highlights_citation",
        "service_citation",
        "has_outdoor_seating_citation",
        "drinks_citation",
        "dietary_restrictions_citation",
        "experience_citation",
        "nutritious_options_citation",
        "creative_menu_citation",
        "has_student_discount_citation",
        "has_senior_discount_citation",
        "local_cuisine_citation",
        "trendy_citation",
        "wheelchair_accessible_citation",
        "noise_level_citation",
        "kids_menu_citation",
        "childrens_activities_citation",
        "if_family_friendly_citation",
        "wait_time_citation",
        "has_live_music_citation",
        "serves_alcohol_citation",
        "michelin_citation",
        "accomodates_large_groups_citation",
        
        # special internal fields
        "_score",
        "_schematization_results",
        
        # outdated location
        "_location",
        "longitude",
        "latitude"
        ]
    
    if field in NOT_USABLE_FIELDS:
        return False
    
    if field.startswith("_score"):
        return False
    
    return True

# a restaurants-specific function dealing with opening_hours
def handle_opening_hours(input_dict):
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def get_order_index(x):
        try:
            return order.index(x['day_of_the_week'])
        except ValueError:
            return len(order)  # or any default value
    
    res = []
    input_dict = sorted(input_dict, key=lambda x: get_order_index(x))
    for i in input_dict:
        res.append(f'open from {i["open_time"]} to {i["close_time"]} on {i["day_of_the_week"]}')
    return res

if __name__ == "__main__":
    print(chunk_text("The text provides general information about the restaurant, including its location in Town and Country shopping center in Palo Alto, its menu offerings such as sushi, rolls, bentos, and sashimi, and its friendly staff. The restaurant has both indoor and outdoor seating, but the indoor seating area is small. The prices are reasonable for the area, and the food is generally fresh and well-prepared. Some reviewers mention that the parking situation can be difficult on weekends.", k=15))
    print(chunk_text(" ", k=15))
