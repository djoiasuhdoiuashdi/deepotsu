from __future__ import print_function
import tensorflow as tf
import bin_networks as net
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training Model Parameters")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="train_model_fusion_11k-log/",
        help="Path to writer logs directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Dataset to use"
    )
    parser.add_argument(
        "--imgtype",
        type=str,
        default="bmp",
        help="Image type"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Overlap of patches"
    )
    parser.add_argument(
        "--multiscale",
        type=bool,
        default=False,
        help="Whether to use multiscale"
    )
    parser.add_argument(
        "--num_block",
        type=int,
        default=6,
        help="The number of blocks"
    )
    return parser.parse_args()

def imshow(img):
    #img = misc.toimage(img, cmin=0, cmax=255)
    plt.imshow(img,cmap='gray')
    #print 'get max value in image is:',np.max(img)
    plt.show()


def im_show_list(imglist):
    #img = misc.toimage(img, cmin=0, cmax=255)
    n_img = len(imglist)
    fig = plt.figure()
    for n in range(n_img):
        fig.add_subplot(1,n_img,n)
        plt.imshow(imglist[n],cmap='gray')

    #print 'get max value in image is:',np.max(img)
    plt.show()

def im_save(name, arr):
    """Save an array to an image file.
    """
    arr = np.uint8(arr)
    im = Image.fromarray(arr)
    im.save(name)
    return
    

def get_image_patch_multiscale(image,imgh,imgw,nimgh,nimgw,overlap=0.1):

    overlap_wid = int(imgw * overlap)
    overlap_hig = int(imgh * overlap)

    height,width = image.shape

    image_list = []
    posit_list = []

    for ys in range(0,height-nimgh,overlap_hig):
        ye = ys + imgh
        for xs in range(0,width-nimgw,overlap_wid):
            xe = xs + nimgw
            imgpath = image[ys:ye,xs:xe]
            imgpath = imgpath.resize((nimgw, nimgh), Image.Resampling.BICUBIC)
            image_list.append(imgpath)
            pos = np.array([ys,xs])
            posit_list.append(pos)

    # last coloum
    for xs in range(0,width-nimgw,overlap_wid):
        xe = xs + nimgw
        ye = height
        ys = ye - nimgh
        imgpath = image[ys:ye,xs:xe]

        imgpath = imgpath.resize((nimgw, nimgh), Image.Resampling.BICUBIC)
        image_list.append(imgpath)
        pos = np.array([ys,xs])
        posit_list.append(pos)

    # last row
    for ys in range(0,height-nimgw,overlap_hig):
        ye = ys + nimgh
        xe = width
        xs = xe - nimgh
        imgpath = image[ys:ye,xs:xe]

        imgpath = imgpath.resize((nimgw, nimgh), Image.Resampling.BICUBIC)
        image_list.append(imgpath)
        pos = np.array([ys,xs])
        posit_list.append(pos)

    # last rectangle
    ye = height
    ys = ye - nimgh
    xe = width
    xs = xe - nimgw
    imgpath = image[ys:ye,xs:xe]

    imgpath = imgpath.resize((nimgw, nimgh), Image.Resampling.BICUBIC)
    image_list.append(imgpath)
    pos = np.array([ys,xs])
    posit_list.append(pos)
    return np.stack(image_list),posit_list


def get_image_patch(image,imgh,imgw,reshape=None,overlap=0.1):

    overlap_wid = int(imgw * overlap)
    overlap_hig = int(imgh * overlap)

    height,width = image.shape

    image_list = []
    posit_list = []

    for ys in range(0,height-imgh,overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        for xs in range(0,width-imgw,overlap_wid):
            xe = xs + imgw
            if xe > width:
                xe = width
            imgpath = image[ys:ye,xs:xe]
            if reshape is not None:
                imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
            image_list.append(imgpath)
            pos = np.array([ys,xs,ye,xe])
            posit_list.append(pos)

    # last coloum
    for xs in range(0,width-imgw,overlap_wid):
        xe = xs + imgw
        if xe > width:
            xe = width
        ye = height
        ys = ye - imgh
        if ys < 0:
            ys = 0

        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)

    # last row
    for ys in range(0,height-imgh,overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        xe = width
        xs = xe - imgw
        if xs < 0:
            xs = 0

        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)

    # last rectangle
    ye = height
    ys = ye - imgh
    if ys < 0:
        ys = 0
    xe = width
    xs = xe - imgw
    if xs < 0:
        xs = 0

    imgpath = image[ys:ye,xs:xe]
    if reshape is not None:
        imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
    image_list.append(imgpath)
    pos = np.array([ys,xs,ye,xe])
    posit_list.append(pos)

    #return np.stack(image_list),posit_list
    return image_list,posit_list


def refinement(patch):
    men = np.mean(patch)
    std = np.std(patch)

    imglist=[]
    #thres = men - 0.2 * std
    thres = men * (1+0.2*((std/128.0)-1))
    res_tmp = patch <= thres
    #imglist.append(res_tmp)
    #imglist.append(patch)

    if np.sum(res_tmp) > 0:
        #print('rescale')
        img_patch = Image.fromarray(patch)
        resized_img = img_patch.resize(patch.shape[:2][::-1], resample=Image.Resampling.BICUBIC)
        patch = np.array(resized_img)

    else:
        patch = (1-res_tmp)*255

    #imglist.append(patch)
    #imshowlist(imglist)
    return patch

def local_thres(patch):
    men = np.mean(patch)
    std = np.std(patch)
    thres = men * (1+0.2*((std/128.0)-1))
    #thres = filters.threshold_otsu(patch)
    mask = patch < thres
    #mask = mask * 255
    return mask

def main():
    imgh = 256
    imgw = 256
    args = parse_args()
    tf.compat.v1.disable_eager_execution()
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, imgh, imgw, 1], name='image')
    imbin = tf.compat.v1.placeholder(tf.float32, shape=[None, imgh, imgw, 1], name='imbin')
    is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')

    nlayers =0
    overlap = args.overlap
    multiscale = args.multiscale
    num_block = args.num_block

    bin_pred_list = net.buildnet(image,num_block,nlayers)

    model_saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    model_logs_dir = args.logs_dir
    print('-'*20)
    print(model_logs_dir)
    ckpt = tf.train.get_checkpoint_state(model_logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model_saver.restore(sess,ckpt.model_checkpoint_path)
        print('*'*20)
        print("Model restored...")
        print('*'*20)

    #image_test_dir = '/mantis/PaperWork/binary/dataset/test/'+FLAGS.dataset+'/'
    image_test_dir = args.dataset


    imagetype = args.imgtype

    if multiscale:
        scalelist = [0.75,1.0,1.25,1.5]
    else:
        scalelist = [1.0]

    for root,sub,images in os.walk(image_test_dir):
        for img in images:
            if not img.endswith(imagetype):
                continue

            if img.startswith('GT'):
                continue

            #if not img.startswith('PR06'):
            #	continue

            print('processing the image:', img)

            image_test = Image.open(os.path.join(image_test_dir, img)).convert("L")
            image_test = np.array(image_test)
            oh,ow = image_test.shape


            res_out = np.zeros((oh,ow))
            num_hit = np.zeros((oh,ow))

            for s in range(num_block):

                #if s <2:
                #	continue

                print('%d-iter is running!'%s)



                for scale in scalelist:

                    crp_w = int(scale*imgw)
                    crp_h = int(scale*imgh)

                    reshape = (imgw,imgh)
                    image_patch,poslist = get_image_patch(image_test,crp_h,crp_w,reshape,overlap=overlap)

                    image_patch = np.stack(image_patch)
                    image_patch = np.expand_dims(image_patch,axis=3)

                    print('scale: %f get patches: %d'%(scale,len(poslist)))

                    npath = len(poslist)

                    batch_size = 10

                    nstep = int( npath / batch_size ) + 1
                    pred_bin_list = None

                    for ns in range(nstep):
                        ps = ns * batch_size
                        pe = ps + batch_size
                        if pe >= npath:
                            pe = npath

                        pathes = image_patch[ps:pe]
                        if pathes.shape[0] == 0:
                            continue

                        feed_dict = {image:pathes,is_training:False}
                        pred_bin = sess.run(bin_pred_list[s],feed_dict=feed_dict)

                        pred_bin = np.squeeze(pred_bin)


                        #print('pred_bin shape:',pred_bin.shape)

                        if ns == 0:
                            pred_bin_list = pred_bin
                        else:
                            #print('ndim is:',pred_bin.ndim,pred_bin.shape)
                            if pred_bin.ndim < 3:
                                pred_bin = np.expand_dims(pred_bin,axis=0)
                            #print('ndim is:',pred_bin.ndim,pred_bin.shape)
                            pred_bin_list = np.concatenate((pred_bin_list,pred_bin),axis=0)

                    print(pred_bin_list.shape,npath,nstep,'*'*20)
                    for n in range(npath):
                        ys = poslist[n][0]
                        xs = poslist[n][1]
                        ye = poslist[n][2]
                        xe = poslist[n][3]

                        re_h = ye - ys
                        re_w = xe - xs


                        if npath == 1:
                            res_path = pred_bin_list
                        else:
                            res_path = pred_bin_list[n]

                        res_path = refinement(res_path)

                        res_path = cv2.resize(res_path.astype('float'), dsize=(re_w,re_h))

                        num_hit[ys:ye,xs:xe] += 1
                        res_out[ys:ye,xs:xe] += res_path

            res_out = res_out / num_hit
            im_save(img[:-4] + '.tiff', res_out)


if __name__ == '__main__':
    main()
