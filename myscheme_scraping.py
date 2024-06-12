from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import json
import os

class MySchemeScraper:
    def __init__(self):
        self.myscheme_url = 'https://rules.myscheme.in'

    def get_scheme_links(self , limit=5):
        driver = webdriver.Firefox()
        driver.get(self.myscheme_url)

        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.ID, "__next")))
        result_elements = driver.find_element('id', '__next').find_element('tag name', 'tbody').find_elements('tag name', 'tr')
        scheme_links = []
        for result_element in result_elements[:limit]:  # Limiting to fetch only 'limit' number of schemes
            table_rows = result_element.find_elements('tag name','td')
            result_details_dict={}
            result_details_dict['sr_no'] = table_rows[0].text
            result_details_dict['scheme_name'] = table_rows[1].text.replace('\nCheck Eligibility','')
            try:
                scheme_link_element = table_rows[2].find_element('tag name', 'a')
                scheme_link = scheme_link_element.get_attribute('href')
                result_details_dict['scheme_link'] = scheme_link
            except (NoSuchElementException, StaleElementReferenceException):
                result_details_dict['scheme_link'] = None
            scheme_links.append(result_details_dict)
            if len(scheme_links) == limit:  # Break the loop if the limit is reached
                break
        driver.close()
        return scheme_links

    def get_scheme_details(self, scheme_links):
        driver = webdriver.Firefox()
        scheme_details = []
        for scheme in scheme_links:
            driver.get(scheme['scheme_link'])
            details = {}
            details['sr_no'] = scheme['sr_no']
            details['scheme_name'] = scheme['scheme_name']
            try:
                details['details'] = driver.find_element(By.ID, 'details').text  # Corrected line
            except NoSuchElementException:
                details['details'] = None
            scheme_details.append(details)
        driver.quit()
        return scheme_details

if __name__=='__main__':
    download_path = os.path.join(os.path.dirname(__file__),'scrapejune.json')
    scraper = MySchemeScraper()
    scraped_scheme_links = scraper.get_scheme_links()
    scraped_scheme_details = scraper.get_scheme_details(scraped_scheme_links)
    json.dump(scraped_scheme_details, open(download_path, 'w'))

