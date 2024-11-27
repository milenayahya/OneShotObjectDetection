from PIL import Image, ImageDraw

# Path to the image
img_path = "MGN/MGN_images/scene50_28.png"

# Bounding boxes in [x, y, w, h] format

bboxes = [
    [
      664.0,
      418.0,
      427.0,
      466.0
    ],
    [
      1076.0,
      248.0,
      339.0,
      388.0
    ]
  ]

# Open the image
image = Image.open(img_path)
draw = ImageDraw.Draw(image)


# Draw the bounding boxes
for bbox in bboxes:
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

image.show()  # Display the image
"""
for bbox in bboxes:
    x1,y1,x2,y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

# Save or display the image
image.show()  # Display the image
# image.save("output_image.png")  # Save the image

"""