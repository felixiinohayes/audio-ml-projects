import requests
from bs4 import BeautifulSoup

def fetch_data(doc_url):
    response = requests.get(doc_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table')
        data = []
        if table:
            elements = table.find_all('p')[3:]
            for index, element in enumerate(elements):
                data.append(element.get_text(strip=True))
                
            data = [data[i:i + 3] for i in range(0, len(data), 3)]
        return data
    
    else:
        raise Exception("Failed to fetch data.")
    
def create_grid(data):

    # Find size of grid
    max_x = max(int(coord[0]) for coord in data)
    max_y = max(int(coord[2]) for coord in data)

    # Fill grid with spaces
    grid = [[' ' for _ in range(max_x+ 1)] for _ in range(max_y + 1)]

    for x, char, y in data:
        grid[max_y - int(y)][int(x)] = char
    
    return grid


def print_grid(grid):
    for row in grid:
        print(''.join(row))


doc_url = 'https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub'

data = fetch_data(doc_url)
grid = create_grid(data)
print_grid(grid)