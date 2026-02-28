import os
import urllib.request
import re

def download_images(url, prefix, count=3):
    print(f"\nFetching from: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch page {url}: {e}")
        return
        
    img_urls = re.findall(r'src="([^"]+)"', html)
    
    valid_urls = []
    for u in img_urls:
        if '/assets/' in u.lower() and (u.lower().endswith('.jpg') or u.lower().endswith('.png') or u.lower().endswith('.jpeg')):
            if u.startswith('/'):
                valid_urls.append('https://dermnetnz.org' + u)
            elif u.startswith('http'):
                valid_urls.append(u)
    
    # Remove duplicates preserving order
    unique_urls = []
    for u in valid_urls:
        if u not in unique_urls:
            unique_urls.append(u)
            
    os.makedirs('data/test_images', exist_ok=True)
    
    downloaded = 0
    for i, img_url in enumerate(unique_urls):
        if downloaded >= count:
            break
        filename = f"data/test_images/{prefix}_{downloaded+1}.jpg"
        print(f"Downloading {filename}...")
        try:
            # Add headers to the image request too
            req_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_img) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

if __name__ == "__main__":
    download_images("https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-benign-melanocytic-lesions", "benign_nevus", 3)
    download_images("https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-melanoma", "malignant_melanoma", 3)
    print("\nDone! Images saved to data/test_images/")
