import urllib.request
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager  
import time


# Grabbing training images by taking pictures of googlestreetview at desired location by coordinate

DRIVER = 'chromedriver'
driver = webdriver.Chrome(ChromeDriverManager().install())


# NYC, lat: ~40, long: ~-73
#Tokyo, lat: ~35, long: ~139

number_of_pictures = 100  # number of pictures driver is grabbing 
latitude = "40.775239"  # starting lat
longitude = "-73.956688"  # starting long
url = 'https://www.google.com/maps/@?api=1&map_action=pano&viewpoint='+latitude+','+longitude+'&heading=-45&pitch=0&fov=80'  # starting image

lat_movement = -0.001  # downward 
long_movement = 0.00001  # left in NYC, right in tokyo

for i in range(number_of_pictures):

	driver.get(url)
	time.sleep(4)  # 4 second wait time from chrome to load
	print(latitude)
	print(longitude)
	screenshot = driver.save_screenshot(latitude + "_" + longitude + ".png")  # saving picture by coordinate label
	latitude = str(round(float(latitude) + lat_movement,6))  # moving down each iter
	longitude = str(round(float(longitude) + long_movement,6))   # moving right each iter 
	url = 'https://www.google.com/maps/@?api=1&map_action=pano&viewpoint='+latitude+','+longitude+'&heading=-45&pitch=0&fov=80'
	print(url) 


driver.quit()

