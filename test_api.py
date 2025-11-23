import requests
import os

def test_image_detection():
    url = "http://127.0.0.1:8001/detect/image"
    files = {'file': open('test_image.jpg', 'rb')}
    try:
        response = requests.post(url, files=files)
        print(f"Image Status: {response.status_code}")
        print(f"Image Response: {response.json()}")
    except Exception as e:
        print(f"Image Test Failed: {e}")

if __name__ == "__main__":
    test_image_detection()
