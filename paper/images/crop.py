from PIL import Image

for img_name in ['2', '4', '6', '9']:
    img = Image.open('{}.png'.format(img_name))
    area = (50, 150, 600, 330)
    cropped = img.crop(area)
    cropped.save('{}_cropped.png'.format(img_name))