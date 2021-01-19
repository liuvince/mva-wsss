import numpy as np
import argparse

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def _crf_with_alpha(cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al

from PIL import Image
import os
from tqdm import tqdm
from utils.utils import mkdir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_la_crf", default="experiment/out_la_crf", type=str)
    parser.add_argument("--out_ha_crf", default="experiment/out_ha_crf", type=str)
    parser.add_argument("--cam_dir", default=None, type=str)
    parser.add_argument("--img_dir", default="data/VOCdevkit/VOC2012/JPEGImages", type=str)

    args = parser.parse_args()
    mkdir(args.out_la_crf)
    mkdir(args.out_ha_crf)

    for id in tqdm(sorted(os.listdir(args.cam_dir))):
        cam_dict = np.load(os.path.join(args.cam_dir, id), allow_pickle=True)[()]
        orig_img = np.asarray(Image.open(os.path.join(args.img_dir, id[:-4] + ".jpg")).convert('RGB'))

        cams = cam_dict['high_res']
        keys = cam_dict["keys"]
        new_cam_dict = {}
        for i, key in enumerate(keys):
            new_cam_dict[key] = cams[i]

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(new_cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, id), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(new_cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, id), crf_ha)