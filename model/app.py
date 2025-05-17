import requests

requerurl = requests.get('https://www.google.com')

print(requerurl.content)
