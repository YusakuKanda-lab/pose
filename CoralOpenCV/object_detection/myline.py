import requests


class LINENotifyBot:
    API_URL = 'https://notify-api.line.me/api/notify'
    def __init__(self, access_token):
        self.__headers = {'Authorization': 'Bearer ' + access_token}

    def send(self, message,image = None):
        payload = {'message': message}
        files = {}
        if image != None :
            files = {'imageFile': open(image, 'rb')}
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
            )


def main():
	l = LINENotifyBot(access_token = '5fr0uMflkLNzcduBXWx42p81OdlEmH73hQ7OuYl5qgM')
	l.send('test',image = "test.jpg")

if __name__ == "__main__":
	main()
