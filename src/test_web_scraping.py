import requests
from bs4 import BeautifulSoup
from datetime import datetime
from utils.log import *

# Search pararmeters
beginYear  = 2023
endYear    = 2024
regionCode = 'US-WA'

# Only download recordings from start_date forward
start_date = datetime(2023, 6, 1) # June 1 2023

save_path = '/Users/giojacuzzi/Downloads'

def get_recording_info(taxonCode):

    search_url = f'https://search.macaulaylibrary.org/catalog?taxonCode={taxonCode}&mediaType=audio&beginYear={beginYear}&endYear={endYear}&regionCode={regionCode}&view=list'
    
    print(f'Retrieving Macaulay Library recording information for {taxonCode}...')
    print(f'Search url: {search_url}')

    response = requests.get(search_url)
    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all recordings (class 'ResultsList-header')
        headers = soup.find_all(class_='ResultsList-header')

        recording_info = []
        for header in headers:
            
            # Extract recording ID
            link = header.find('a', target='_blank')
            if link:
                href = link['href']
                id = href.split('/')[-1]
            else:
                id = None

            # Extract recording date
            meta = header.find_next_sibling(class_='ResultsList-meta')
            date_tag = meta.find('time', datetime=True)
            if date_tag:
                date = date_tag['datetime']
                date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
            else:
                date = None

            if date >= start_date:
                recording_info.append({'id': id, 'date': date})
            else:
                print_warning(f'Recording {id} made prior to start_date ({date}). Skipping...')
        
        return recording_info
    else:
        # If the request was not successful, print an error message
        print_error("Unable to fetch data from the website")
        return []

# Example usage
taxonCode = 'grhowl'
recording_info = get_recording_info(taxonCode)

if recording_info:
    print("Downloading recordings for search query '{}'".format(taxonCode))

    # Download each recording
    for info in recording_info[0:4]:
        print(f'Downloading ML{info["id"]} ({info["date"]})...')
        # download_url = f'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{id}'
        # response = requests.get(download_url, stream=True)
        # if response.status_code == 200: # If request was successful
        #     with open(f'{save_path}/{taxonCode}-ML{info["id"]}.mp3', 'wb') as f: # Open file for writing
        #         for chunk in response.iter_content(chunk_size=1024): # Write response data to file
        #             f.write(chunk)
        #     print_success("Download successful.")
        # else:
        #     print_error("Unable to download the file.")
else:
    print("No recording IDs found for taxon code '{}'".format(taxonCode))
