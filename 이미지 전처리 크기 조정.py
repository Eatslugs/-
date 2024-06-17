import os
import cv2
import numpy as np

mother_folder = "D:\\smooth"
daughter_folder = "D:\\smooth_final"
filenames = os.listdir(mother_folder)
a= len(filenames)

num=0
for name in filenames:
    num+=1
    try:
        img = cv2.imread(mother_folder+'\\'+name)
        h, w = img.shape[:2]
        
        # 목표 크기
        target_h, target_w = 200,200
        
        # 비율 계산
        scale = min(target_w / w, target_h / h)
        
        # 새로운 크기 계산
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 이미지 크기 조정
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 빈 캔버스 생성
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 이미지 붙여넣기 (중앙 정렬)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        # 이미지 저장
        cv2.imwrite(daughter_folder+'\\'+name, padded_img)
        print(name,'done')
    except:
        print('error at',name)
    print(f'{num} out of {a}')