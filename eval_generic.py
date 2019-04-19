from glob import glob
from os.path import basename
import argparse

def test(gts, test_folder, threshold, skip):
    dets_txt = glob(test_folder + '/*')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for dets_name in dets_txt:
        line = open(dets_name).readline()
        if len(line) == 0:
            continue
        #print line
        carplate, score = line.split()[:2]
        carplate = carplate.strip()
        score = float(score)
        carplate = carplate.decode('utf8')
        carplate = carplate[skip:]
        name = basename(dets_name)
        gt = gts[name]
        #print gt, len(gt), carplate, len(carplate)
        if score > threshold:
            if carplate== gt: #.encode('utf8'):
                tp += 1
            else:
                fp += 1
    #fn = len(gts) - tp - fp
    fn = len(dets_txt) - tp - fp
    if tp +fp ==0:
        prec =0
    else:
        prec = float(tp) / float(tp + fp)
    if tp +fn ==0:
        recall=0
    else:
        recall = float(tp) / float(tp + fn)
    return prec, recall


def load_gt(ocrtxt_file, skip):
    gts ={}
    for line in open(ocrtxt_file, 'r'):
        fname, label = line.split(';')
        bname = basename(fname).replace('_plate.jpg', '')

        plate = label.strip().decode('utf8')
        plate = plate.replace('|', '')
        #print bname, plate
        gts[bname] = plate[skip:]
    print "Total number of gt", len(gts)
    return gts

def eval(ocrlabel, res_dir, skip):
    gts = load_gt(ocrlabel, skip)
    # print gts
    tmax = 10.0
    tmin = 0.0
    tnow = 0.0
    while True:
        prec, recall = test(gts, res_dir, tnow, skip)
        if abs(recall - 0.9) < 0.001:
            break
        if recall > 0.9:
            tmin = tnow
            tnow = (tnow + tmax) * 0.5
        if recall < 0.9:
            tmax = tnow
            tnow = (tnow + tmin) * 0.5
        if abs(tmax - tmin) < 0.001:
            break
        print(tnow, test(gts, res_dir, tnow, skip))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation OCR results')
    parser.add_argument('--ocrlabel', default='/ssd/wfei/data/plate_for_label/energy_cars/energy_plates_ocrlabel_test_217.txt',
                        type=str, help='Output plate label dir')
    parser.add_argument('--res_dir', default='/ssd/wfei/data/plate_for_label/energy_cars/20190124/results/k11_energy_caffek11',
    #parser.add_argument('--res_dir', default='k11_caffe_v1.1',
                        type=str, help='Input test image dir')
    parser.add_argument('--skip', default=0, type=int, help='Skip any characters in evaluation')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    skip =0
    print "Skip {} characters for evaluation!!!".format(skip)

    #res_base = '/ssd/wfei/data/benchmark/k11/'
    #res_dir = join(res_base, args.res_dir)
    eval(args.ocrlabel, args.res_dir, args.skip)
