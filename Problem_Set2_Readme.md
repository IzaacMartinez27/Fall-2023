##  Problem Set 2

# 1.) Load an RGB image of your choice from a URL, Resize the image to 224x224, Show a grayscale copy
```Python
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt



# Define the URL of the image
image_url = 'https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRcbLjcZKWWHRRpf5gdOSCI78jLz3gpNgL67AcTD3zFE-zU_GTG'  # Replace with the URL of your image

# Download the RGB image from the URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Resize the image to 224x224
image_resized = image.resize((224, 224))

# Convert the resized image to grayscale
image_grayscale = image_resized.convert('L')

# Show the grayscale image
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.title('Original Image (Resized)')
plt.imshow(image_resized)
plt.axis('off')

plt.subplot(122)
plt.title('Grayscale Image')
plt.imshow(image_grayscale, cmap='gray')
plt.axis('off')

plt.show()
```
# 2.) Convolve with 10 random filters and show filters and features maps for each
```Python
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import cv2
from IPython.display import display, Image as IPImage, Markdown

# Define the URL of the image
image_url = 'https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRcbLjcZKWWHRRpf5gdOSCI78jLz3gpNgL67AcTD3zFE-zU_GTG'  # Replace with the URL of your image

# Download the image from the URL
response = requests.get(image_url, stream=True)
image = Image.open(response.raw).convert('L')  # Convert to grayscale

# Define the number of random filters
num_filters = 10

# Initialize an empty list to store filter images
filter_images = []

# Initialize an empty list to store feature maps
feature_maps = []

# Generate random filters and apply convolution
for i in range(num_filters):
    random_filter = np.random.randn(3, 3)
    filter_images.append(random_filter)

    feature_map = cv2.filter2D(np.array(image), -1, random_filter)
    feature_maps.append(feature_map)

# Display filter images
for i in range(num_filters):
    filter_image = filter_images[i]
    plt.imshow(filter_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Filter {i + 1}')
    plt.show()

# Display feature maps
for i in range(num_filters):
    feature_map = feature_maps[i]
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')
    plt.title(f'Feature Map {i + 1}')
    plt.show()
```
