from bs4 import BeautifulSoup
import requests


def understand_report(search_term,source_lang,target_lang='hi'):

    url = 'https://api.mymemory.translated.net/get'
    params = {
        'q': search_term,
        'langpair': f'{source_lang}|{target_lang}'
    }

    response = requests.get(url, params=params)
    if response.ok:
        data = response.json()
        if data['responseStatus'] == 200:
            translation = data['responseData']['translatedText']
          #  print('Arabic Translation:', translation)
            return translation
        else:
            print('Error:', data['responseDetails'])
            msg='Error:'+ data['responseDetails']
            return msg
    else:
        print('Error:', response.status_code)
        msg='Error:'+ response.status_code
        return msg
    ##################################################################

# search_term= "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine",
# source_lang = 'en'
# target_lang = 'hi'
# full_report=understand_words(search_term,source_lang,target_lang)
# print(full_report)


