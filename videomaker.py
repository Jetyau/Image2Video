import cv2
import os
from glob import glob
import argparse
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import re


def main(args):
    path1 = args.path1
    exp_name1 = os.path.basename(path1)
    path2 = args.path2
    exp_name2 = os.path.basename(path2)
    gt_path = args.gt_path
    img_path = args.img_path
    save_path = './' + exp_name1 + 'vs' + exp_name2 + '_video.avi'
    video_writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'), 0.5, (256 *2, 256 * 2))

    exp1_pred_dir = os.listdir(path1)
    exp1_pred_dir = [exp1_pred_dir[i] for i in range(len(exp1_pred_dir)) if re.search('^[0-9]',exp1_pred_dir[i]) is not None]
    if len(exp1_pred_dir) > 1:
        raise NotImplementedError
    exp1_pred_path = os.path.join(path1, exp1_pred_dir[0])
    exp1_pred_imgs = glob(exp1_pred_path + '/*.png')

    exp2_pred_dir = os.listdir(path2)
    exp2_pred_dir = [exp2_pred_dir[i] for i in range(len(exp2_pred_dir)) if re.search('^[0-9]',exp2_pred_dir[i]) is not None]
    if len(exp2_pred_dir) > 1:
        raise NotImplementedError
    exp2_pred_path = os.path.join(path2, exp2_pred_dir[0])
    exp2_pred_imgs = glob(exp2_pred_path + '/*.png')

    exp1_pred_imgs = sorted(exp1_pred_imgs, key= os.path.basename)
    exp2_pred_imgs = sorted(exp2_pred_imgs, key= os.path.basename)
    assert len(exp1_pred_imgs) == len(exp2_pred_imgs)

    transform = Compose([
        transforms.Resize(256, 256),
    ])

    for _ in range(len(exp1_pred_imgs)):
        assert os.path.basename(exp1_pred_imgs[_]) == os.path.basename(exp2_pred_imgs[_])
        current_case = os.path.basename(exp1_pred_imgs[_])[os.path.basename(exp1_pred_imgs[_]).find('case'):os.path.basename(exp1_pred_imgs[_]).find('_')]
        current_frame_num = os.path.basename(exp1_pred_imgs[_])[os.path.basename(exp1_pred_imgs[_]).find('_'):os.path.basename(exp1_pred_imgs[_]).rfind('.')]
        corresponding_img = os.path.join(img_path, current_case  + current_frame_num +'.png')
        corresponding_gt = os.path.join(gt_path, current_case  + current_frame_num +'.png')
        try:
            print(corresponding_img)
            corresponding_img = cv2.imread(corresponding_img)
            corresponding_gt = cv2.imread(corresponding_gt)
        except FileNotFoundError:
            exit()
        
        augmented = transform(image=corresponding_img, mask=corresponding_gt)
        corresponding_img = augmented['image']
        corresponding_gt = augmented['mask']
        exp1_pred = transform(image=cv2.imread(exp1_pred_imgs[_]))['image']
        exp1_pred = cv2.putText(exp1_pred, 'exp1', (0,0), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255))
        exp2_pred = transform(image=cv2.imread(exp2_pred_imgs[_]))['image']
        #exp2_pred = cv2.putText(exp2_pred, 'exp2', (0,0), cv2.FONT_HERSHEY_SIMPLEX, 50, (0, 0, 255))
        img_ground = cv2.hconcat([corresponding_img, corresponding_gt])
        img_preds = cv2.hconcat([exp1_pred, exp2_pred])
        whole_img_frame = cv2.vconcat([img_ground, img_preds])
        video_writer.write(whole_img_frame)
    video_writer.release()


    




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', required = True, type=str, help='exp name 1')
    parser.add_argument('--path2', required = True, type=str, help='exp name 2')
    parser.add_argument('--gt_path', required = True, type=str, help='path to gt')
    parser.add_argument('--img_path', required = True, type=str, help='path to original imgs')
    args = parser.parse_args()
    main(args)