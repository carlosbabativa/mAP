# Testing and graphing mAP AlexeyAB/darknet trained models
#   based on cartucho/mAP testing repo
# This script automates the process of preparing the darknet model and data
#   to be tested and graphed with mAP
import os
import shutil
import argparse

def jp(*args):
    return os.path.join(*args)

def main(args):
    # parse arguments from weights file
    cfg     = 'cfg/{}.cfg'.format(args.config)
    weights = 'backup/{}_best.weights'.format(args.config)
    data    = 'data/{}.data'.format(args.ds_name)
    ds_name = args.ds_name
    ds_path = os.path.join(args.dss_path,ds_name)
    d_valid = os.path.join(ds_path,args.d_valid)
    d_train = os.path.join(ds_path,'data_train')
    dn_path = args.dnet_path
    txt_pth = jp(os.path.expanduser('~'),'test/{}.txt'.format(ds_name))
    mAP_pth = args.map_path
    mAP_xt  = jp(mAP_pth,'scripts/extra')
    mAP_in_pth = jp(mAP_pth,'input')
    mAP_in  = {
        'dr':'detection-results',
        'gt':'ground-truth',
        'im':'images',
        'io':'images-optional'
        }
    f_mAP_in = {k:jp(mAP_in_pth,v) for k,v in mAP_in.items()}
    # 0.1 List data_val and obtain darknet's extended results.txt file
    '''
        ls -d ~/datasets/ds_name/data_val/*.jpg > ~/test/thing.txt
        ./darknet detector test cfg/ds_name.data cfg/yolov3-bs_land_gen_test.cfg backup/yolov3-bs_land_gen_final.weights -ext_output -dont_show < ~/test/ds_name.txt > ~/test/results-thing.txt
    '''
    list_val_imgs = 'ls -d {}/*.jpg >{}'.format(d_valid,txt_pth)
    os.system(list_val_imgs)
    results = txt_pth.replace('.txt','-results.txt')
    os.chdir(dn_path)
    get_dnet_results  = './darknet detector test {} {} {} -ext_output -dont_show < {} > {}'.format(data,cfg,weights,txt_pth,results)
    os.system(get_dnet_results)
    shutil.copyfile(results,jp(mAP_xt,'result.txt'))

    # 0.2 populate mAP/input
    os.chdir(mAP_pth)
    if os.path.exists( mAP_in_pth ):
        inp = r'{}'.format(mAP_in_pth)
        inp_old = inp+'_old'
        if os.path.exists(inp_old):
            shutil.rmtree(inp_old)
        os.rename( inp, inp_old )
        os.mkdir(inp)
    names = list(filter(lambda f: '.names' in f, os.listdir(ds_path)))[0]
    shutil.copyfile(jp(ds_path,names),jp(mAP_xt,'class_list.txt'))
    # images = list(filter(lambda f: '.jpg' in f or 'boxes' in f, os.listdir(d_valid)))
    # labels = list(filter(lambda f: '.txt' in f or 'boxes' in f or '.xml' in f, os.listdir(d_valid)))
    shutil.copytree(d_valid,f_mAP_in['gt'],ignore=shutil.ignore_patterns('*.jpg','*.xml','boxes'))
    shutil.copytree(d_valid,f_mAP_in['im'],ignore=shutil.ignore_patterns('*.txt','*.xml','boxes'))
    shutil.copytree(d_valid,f_mAP_in['io'],ignore=shutil.ignore_patterns('*.txt','*.xml','boxes'))
        
    #   cartucho/mAP steps:
    #1. Create the ground-truth files
    #2. Copy the ground-truth files into the folder input/ground-truth/
    #3. Create the detection-results files
    #4. Copy the detection-results files into the folder input/detection-results/
    os.mkdir(f_mAP_in['dr'])
    os.system('python3 {}'.format(jp(mAP_xt,'cvt_yolo.py')))
    
    #5. Run the code: python main.py
    '''
        python3 ~/GitHub/mAP/main.py --set-class-iou thing 0.1
    '''
    #TODO: capture iou as arg, iterate over class names 
    os.system('python3 {} -o {} --set-class-iou thing 0.1'.format(jp(mAP_pth,'main.py'),'output_'+cfg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--validation_dir',
        dest='d_valid',
        default='data_val',
        help='Name of directory within dataset directory with validation images to process',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        dest='ds_name',
        # default='data',
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    parser.add_argument(
        '-p',
        '--datasets-path',
        dest='dss_path',
        default='~/datasets',
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    parser.add_argument(
        '-cfg',
        '--config',
        dest='config',
        help='model configuration ID',
    )
    parser.add_argument(
        '-dn',
        '--dnet_path',
        dest='dnet_path',
        default='~/software/darknet_',
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    parser.add_argument(
        '-m',
        '--map_path',
        dest='map_path',
        default='~/GitHub/mAP',
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    
    args = parser.parse_args()
    args.__dict__ = {k:arg.replace('~',os.path.expanduser('~')) for k,arg in args.__dict__.items()}
    main(args)
