from PIL import Image
import numpy as np
import math
import os

def cutline(image,x_para=7000,y_para=7000):
    img=np.array(image)
    x_max = x_min = image.size[0]//2
    y_max = y_min = image.size[1]//2
    while np.sum(img[y_max,:]) > y_para and y_max < image.size[1]-1:
        y_max+=1
    while np.sum(img[y_min,:]) > y_para and y_min > 1:
        y_min-=1
    while np.sum(img[:,x_max]) > x_para and x_max < image.size[0]-1:
        x_max+=1
    while np.sum(img[:,x_min]) > x_para and x_min > 1:
        x_min-=1

    return x_min, y_min, x_max, y_max

mother_folder = "C:\\Users\\Administrator\\Desktop\\research\\galaxy data\\images\\train_spiral"
filenames = os.listdir(mother_folder)

num=0
for name in filenames:
    num+=1
    try:
        image = Image.open(mother_folder + '\\' + name)

        x_min, y_min, x_max, y_max = cutline(image)
        image_rotate_temp = image.rotate(-math.degrees(math.atan((y_max-y_min)/(x_max-x_min))))

        image_cropped_temp = image_rotate_temp.crop(cutline(image_rotate_temp))

        temp = cutline(image_cropped_temp,10000,10000)
        height_ori = temp[3] - temp[1]
        turn=0

        for i in range(-30,30):
            temp = cutline(image_cropped_temp.rotate(3*i),10000,10000)
            if temp[3] - temp[1] < height_ori:
                height_ori = temp[3] - temp[1]
                turn = 3*i

        img_rotate = image_rotate_temp.rotate(turn)

        x_min, y_min, x_max, y_max = cutline(img_rotate,y_para=11000)

        img_cropped = img_rotate.crop((x_min-10, y_min, x_max+10, y_max))

        w,h = img_cropped.size
        cen_bright = sum(np.array(img_cropped)[h//2][w//2])
        problem_pixel = 0
        image_ok = True
        for x,y in [(x,y) for x in range(w) for y in range(h)]:
            if sum(np.array(img_cropped)[y][x]) > cen_bright and ((x-w/2)**2+(y-h/2)**2)**0.5 > 30:
                problem_pixel+=1
                if problem_pixel > 80:
                    image_ok = False
                    break

        if image_ok == False or img_cropped.size[0]<img_cropped.size[1]:
            img_cropped.save("C:\\Users\\Administrator\\Desktop\\research\\galaxy data\\images\\trash data\\"+name)
            print(name,'was disgarded')
        else:
            img_cropped.save("C:\\Users\\Administrator\\Desktop\\research\\galaxy data\\images\\train_spiral_cut\\"+name)
            print(name + ' was saved')
    except:
        print('error at '+name)
    print(f'{num} out of {len(filenames)} images done')