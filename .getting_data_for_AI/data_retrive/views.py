from django.http import JsonResponse
from django.views import View
import requests
from bs4 import BeautifulSoup
import re
from datetime import timedelta, date
from .models import CarData

urls = [
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/honda/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/mitsubishi/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/subaru/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/mercedes-benz/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/hyundai/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/nissan/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/lexus/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/toyota/',
    'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/ford/',
]

def scrape_data():
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        hrefs = soup.find_all('a', class_='mask')
        advs = re.findall(r'href=[\'"]?([^\'" >]+)', str(hrefs))
        base_url = 'https://www.unegui.mn'
        links = list(map(lambda x: base_url + x, advs))

        for link in links:
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser')

            try:
                # Extracting posted date
                posted_date_text = soup.find('span', class_='date-meta').text.strip()
                if 'Өнөөдөр' in posted_date_text:
                    posted_date = date.today()
                elif 'Өчигдөр' in posted_date_text:
                    posted_date = date.today() - timedelta(days=1)
                else:
                    # Extract only the date part from the string
                    match = re.search(r'\d{4}-\d{2}-\d{2}', posted_date_text)
                    if match:
                        date_string = match.group()
                        posted_date = date.fromisoformat(date_string)
                    else:
                        # Handle the case when the date string format does not match
                        # Use a default date or log an error
                        posted_date = None
                        
                manufacturer = soup.find('h1', class_='title-announcement').text.strip().split(" ")[0]
                model = ' '.join(re.findall(r'^(.+?),', soup.find('h1', class_='title-announcement').text.strip())).split(' ', 1)[1]

                engine_capacity_element = soup.select_one('ul.chars-column li:nth-of-type(1) a.value-chars')
                engine_capacity = engine_capacity_element.text.strip() if engine_capacity_element else "Unknown"

                transmission_element = soup.select_one('ul.chars-column li:nth-of-type(2) a.value-chars')
                transmission = transmission_element.text.strip() if transmission_element else None

                steering_element = soup.select_one('ul.chars-column li:nth-of-type(3) a.value-chars')
                steering = steering_element.text.strip() if steering_element else None

                type_element = soup.select_one('ul.chars-column li:nth-of-type(4) a.value-chars')
                type = type_element.text.strip() if type_element else None

                color_element = soup.select_one('ul.chars-column li:nth-of-type(5) a.value-chars')
                color = color_element.text.strip() if color_element else None

                manufacture_year_element = soup.select_one('ul.chars-column li:nth-of-type(6) a.value-chars')
                manufacture_year = int(manufacture_year_element.text.strip()) if manufacture_year_element else None

                import_year_element = soup.select_one('ul.chars-column li:nth-of-type(7) a.value-chars')
                import_year = int(import_year_element.text.strip()) if import_year_element else None

                drive_element = soup.select_one('ul.chars-column li:nth-of-type(8) a.value-chars')
                drive = drive_element.text.strip() if drive_element else None

                interior_color_element = soup.select_one('ul.chars-column li:nth-of-type(9) span.value-chars:nth-of-type(2)')
                interior_color = interior_color_element.text.strip() if interior_color_element else None


                drive_type_element = soup.select_one('ul.chars-column li:nth-of-type(11) a.value-chars')
                drive_type = drive_type_element.text.strip() if drive_type_element else None

                mileage_element = soup.select_one('ul.chars-column li:nth-of-type(12) a.value-chars')
                mileage = mileage_element.text.strip() if mileage_element else None


                price_text = soup.select_one('div.announcement-price__cost meta:nth-of-type(2)')
                price = int(re.findall(r'[0-9]+', str(price_text))[0]) if price_text else None

                unique_id = soup.find('span', itemprop="sku").text.strip()

                # Saving data to the database
                if posted_date and manufacturer and model and price is not None:
                    car_data, created = CarData.objects.update_or_create(
                        unique_id=unique_id,
                        defaults={
                            'manufacturer': manufacturer,
                            'model': model,
                            'posted_date': posted_date,
                            'engine_capacity': engine_capacity,
                            'transmission': transmission,
                            'steering': steering,
                            'type': type,
                            'color': color,
                            'manufacture_year': manufacture_year,
                            'import_year': import_year,
                            'drive': drive,
                            'interior_color': interior_color,
                            'drive_type': drive_type,
                            'mileage': mileage,
                            'price': price,
                        }
                    )
            except Exception as e:
                print(f"An error occurred while processing {link}: {e}")

class ScrapeView(View):
    def get(self, request, *args, **kwargs):
        try:
            scrape_data()
            return JsonResponse({"status": "success"}, status=200)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    