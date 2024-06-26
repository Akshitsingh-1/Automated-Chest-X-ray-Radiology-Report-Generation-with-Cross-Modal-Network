import requests
from bs4 import BeautifulSoup



def split(str):
    # Split the string into individual words
    words = str.split()

    # Add a "+" after each word
    new_str = '+'.join(words)

    # print(new_str)
    return new_str
def understand_words(search_term,source_lang,target_lang):
    all_possible_meaing = []

    # Add a new item to the end of the list using append()
    url = "https://context.reverso.net/translation/"+source_lang+"-"+target_lang+"/"

    # Define the URL to search and the search term
   # url = "https://context.reverso.net/translation/english-arabic/"
   # url = f'https://context.reverso.net/translation/{source_lang}-{target_lang}/{sentence}'
    search_term = split(search_term)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.post(url + search_term, headers=headers)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the Arabic translation results on the webpage
    translations = soup.find_all('a', {'class': 'translation'})

    # Print the Arabic translation results
    # print("Arabic translations of '{}':".format(search_term))
    for translation in translations:
        #print(translation.text.strip())
        all_possible_meaing.append(translation.text.strip()[:].split(" ")[0])

    return all_possible_meaing


# source_lang = 'english'
# target_lang = 'Hindi'

# # search_term = "The heart size and pulmonary vascularity appear within normal limits." \
# #               " A large hiatal hernia is noted. The lungs are free of focal airspace disease. " \
# #               "No pneumothorax or pleural effusion is seen." \
# #               " Degenerative changes are present in the spine"

# search_term = "clear"
# all_possible_meaing=understand_words(search_term,source_lang,target_lang)
# del all_possible_meaing[0]
# print(all_possible_meaing)

