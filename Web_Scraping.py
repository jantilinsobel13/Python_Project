import requests
from bs4 import BeautifulSoup

url = "https://veloriatech.com/"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

print("Veloria Tech Links:")
for link in soup.find_all("a", href=True):
    text = link.get_text(strip=True)
    if text:  
        print("-", text, "->", link["href"])
