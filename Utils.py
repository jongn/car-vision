import time
import requests

def grab_image(url, img_name):
	path = "Images/" + img_name
	img_data = requests.get(url).content
	with open(path, 'wb') as handler:
		handler.write(img_data)

def grab_image_1min(url, img_name):
	frame = 0
	while True:
		image_name = str(frame) + "_" + str(frame) + "_" + img_name
		grab_image(url, image_name)
		time.sleep(60)
		frame = frame + 1

def main():
	url = "http://www.dot.ca.gov/cwwp2/data/d4/cctv/image/TV388_N1PRESIDIO.jpg?1378316522948"
	img_name = "TV388_N1PRESIDIO.jpg"
	grab_image_1min(url, img_name)



if __name__ == "__main__":
	main()