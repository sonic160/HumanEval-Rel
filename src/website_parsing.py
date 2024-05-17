import requests
from bs4 import BeautifulSoup
import re
from urllib.request import urlopen, Request
import time
from tqdm import tqdm


def extract_text_html(html):
    html_decoded = html.decode('utf-8')
    text_list = re.findall(r'>(.*?)<', html_decoded)
    print(text_list)
    return text_list

def list_link(url, links):
    # lien de la page à scrapper
    reponse = requests.get(url)
    page = reponse.content
    # transforme (parse) le HTML en objet BeautifulSoup
    soup = BeautifulSoup(page, "html.parser")
    # récupération de tous les titres 
    tags_a = soup.find_all('a')
    # Parser les liens
    for tag in tags_a:
        link = tag.get('href')
        if link and link.startswith("https://help.reliasoft.com") : # and link.endswith("html"):
            links.append(link)
    return links
        

def scrapping():
    data = []
    links = ["https://help.reliasoft.com/reference/system_analysis/sa/basics_of_system_reliability_analysis.html",
             "https://help.reliasoft.com/reference/life_data_analysis/lda/introduction_to_life_data_analysis.html",
             "https://help.reliasoft.com/reference/accelerated_life_testing_data_analysis/alt/introduction_to_accelerated_life_testing.html",
             "https://help.reliasoft.com/reference/reliability_growth_and_repairable_system_analysis/rg_rsa/rga_overview.html",
             "https://help.reliasoft.com/reference/experiment_design_and_analysis/doe/doe_overview.html",
             ]
    
    url = "https://www.reliawiki.com/index.php/Main_Page"
    #Add others links to the list
    links = list_link(url, links)

    for i in tqdm(range(0,20), desc = 'Scrapping'):
        #print('test')
        headers = {'User-Agent': 'Mozilla/5.0'}
        request = Request(links[i], headers=headers)
        html = urlopen(request).read()
        soup = BeautifulSoup(html, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()
        data += text.split('\n')

        time.sleep(1)

        #add others links that we could find in the link we already have
        links = list_link(url, links)

    clean_data = [element.strip() for element in data if element.strip()]
    print(clean_data)


scrapping()