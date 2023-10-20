import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("/content/gutters1.JPG", cv2.IMREAD_GRAYSCALE)

# Split the grayscale image into its planes
split_img = cv2.split(img)
final_img = []  # Image Array to store the final images

for plane in split_img:
    # Apply dilation to expand the image (convolving with a kernel)
    exp = cv2.dilate(plane, np.ones((5, 5), np.uint8))

    # Apply Gaussian blur to reduce noise and blur the image
    dst = cv2.GaussianBlur(exp, (3, 3), 0)

    # Remove shadows
    shad = 255 - cv2.absdiff(plane, dst)

    # Normalize the image
    norm = cv2.normalize(shad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    final_img.append(norm)

# Merge all planes to form a multicolor image
result = cv2.merge(final_img)

# Convert the result to RGB
result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

# Clip pixel values to the valid range for displaying as RGB
result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

# Save the processed image
cv2.imwrite("cleaned-gutter.jpg", result_rgb)

# Display the final processed image
plt.imshow(result_rgb)
plt.title("Processed Image")
plt.axis("off")
plt.show()
