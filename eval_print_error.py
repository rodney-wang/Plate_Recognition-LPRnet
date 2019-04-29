from glob import glob
from os.path import join, exists, basename, splitext
import os
import sys
import shutil
import argparse
import pdb

def load_gt(ocrtxt_file, skip):
    gts, gts_fpath ={}, {}
    for line in open(ocrtxt_file, 'r'):
        fname, label = line.split(';')
        bname = basename(fname).replace('_plate.jpg', '')

        plate = label.strip().decode('utf8')
        plate = plate.replace('|', '')
        #print bname, plate
        gts[bname] = plate[skip:]
        gts_fpath[bname] = fname

    print "Total number of gt", len(gts)
    return gts, gts_fpath

def eval_and_write_error(test_folder, ocrlabel, error_dir, threshold, skip):

    gts, gts_fpath = load_gt(ocrlabel, skip)
    dets_txt = glob(join(test_folder, '*'))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if not exists(error_dir):
        os.makedirs(error_dir)
    for dets_name in dets_txt:
        line = open(dets_name).readline().decode('utf8')
        if len(line) == 0:
            continue
        carplate, score = line.split()[:2]
        score = float(score)
        carplate = carplate[skip:]
        #name = basename(splitext(dets_name)[0])
        name = basename(dets_name)
        #pdb.set_trace()
        if name not in gts:
            continue
        gt = gts[name]
        if score > threshold:
            if carplate == gt:
                tp += 1
            else:
                fp += 1
                print name, gt, carplate
                src_f = gts_fpath[name]
                tgt_f = join(error_dir, name+'_'+ gt+ '_'+carplate+'.jpg')
                print(src_f, tgt_f)
                shutil.copyfile(src_f, tgt_f)
    fn = len(dets_txt) - tp - fp
    prec = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    return prec, recall


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation OCR results')
    parser.add_argument('--ocrlabel', default='/ssd/wfei/data/plate_for_label/k11_entrance/20190424_k11_entrance_ocrlabel.txt',
                        type=str, help='Output plate label dir')
    parser.add_argument('--res_dir', default='/ssd/wfei/data/plate_for_label/k11_entrance/results',
                        type=str, help='Input test image dir')
    parser.add_argument('--error_dir', default='/ssd/wfei/data/plate_for_label/k11_entrance/errors',
                        type=str, help='Input test image dir')
    parser.add_argument('--thresh', default=3.82, type=float, help='Threshhold for computation of precision recall')
    parser.add_argument('--skip', default=0, type=int, help='Skip any characters in evaluation')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    prec, recall = eval_and_write_error(args.res_dir, args.ocrlabel, args.error_dir, args.thresh, args.skip)
    print('Precision = {}, Recall = {}'.format(prec, recall))
