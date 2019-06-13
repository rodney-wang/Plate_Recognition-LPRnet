#!/usr/bin/env python
# coding=utf-8
import os
import glob
import codecs

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z','_'
         ]
dict = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
        'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}


def fname_to_ocrtxt(ocrtxt_file, plate_dir):

    img_files =  glob.glob(os.path.join(plate_dir, '*.jpg'))
    fo = codecs.open(ocrtxt_file, "w", encoding='utf-8')
    for i, fname in enumerate(img_files):
        print '### Processing ', i, fname
        bnames = os.path.basename(fname).split('_')
        prov, rest = bnames[0], bnames[1]
        if prov not in dict:
            continue
        ocr = dict.get(prov) + rest
        print ocr

        plate_label = '|' + '|'.join(ocr.decode('utf8')) + '|'
        line = fname + ';' + plate_label + '\n'
        print line
        fo.write("%s" % line)

    print "Plate file name converted to ocrlabel file", ocrtxt_file


if __name__ == '__main__':

    img_dir = '/mnt/soulfs2/wfei/data/ocr_training/lprraw/plates'
    img_dir = '../train/'
    #ocr_txt = '/mnt/soulfs2/wfei/data/ocr_training/lprraw/lpr_github_plates_ocrlabel.txt'
    ocr_txt = 'lpr_github_plates_ocrlabel.txt'
    fname_to_ocrtxt(ocr_txt, img_dir)
