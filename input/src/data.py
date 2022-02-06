"""Downloads data

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import requests
import os
import tqdm

class GoogleDriveDownloader(object):
    """
    Downloading a file stored on Google Drive by its URL.
    If the link is pointing to another resource, the redirect chain is being expanded.
    Returns the output path.
    """
    
    base_url = 'https://docs.google.com/uc?export=download'
    chunk_size = 32768
    
    def __init__(self, url, out_dir):
        super().__init__()
        
        self.out_name = url.rsplit('/', 1)[-1]
        self.url = self._get_redirect_url(url)
        self.out_dir = out_dir
    
    @staticmethod
    def _get_redirect_url(url):
        response = requests.get(url)
        if response.url != url and response.url is not None:
            redirect_url = response.url
            return redirect_url
        else:
            return url
    
    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def _save_response_content(self, response):
        with open(self.fpath, 'wb') as f:
            bar = tqdm.tqdm(total=None)
            progress = 0
            for chunk in response.iter_content(self.chunk_size):
                if chunk:
                    f.write(chunk)
                    progress += len(chunk)
                    bar.update(progress - bar.n)
            bar.close()
    
    @property
    def file_id(self):
        return self.url.split('?')[0].split('/')[-2]
    
    @property
    def fpath(self):
        return os.path.join(self.out_dir, self.out_name)
    
    def download(self):
        os.makedirs(self.out_dir, exist_ok=True)
        
        if os.path.isfile(self.fpath):
            print('File is downloaded yet:', self.fpath)
        else:
            session = requests.Session()
            response = session.get(self.base_url, params={'id': self.file_id}, stream=True)
            token = self._get_confirm_token(response)

            if token:
                response = session.get(self.base_url, params={'id': self.file_id, 'confirm': token}, stream=True)
            else:
                raise RuntimeError()

            self._save_response_content(response)
        
        return self.fpath


def main():
	os.makedirs("../data/", exist_ok=True)
	url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	dl = GoogleDriveDownloader(url, '../data/')
	dl.download()

	url_b = 'https://drive.google.com/file/d/1MVfYF0qVgKHTQKFdA7lexGWnRIs7Ax9c/view?usp=sharing'
	dl_b = GoogleDriveDownloader(url_b, '../data/')
	dl_b.download()

	os.system("unzip ../data/birds.zip -d ../data/")
	os.system("tar -xvf ../data/CUB_200_2011.tgz -C ../data/")



if __name__ == '__main__':
	main()
