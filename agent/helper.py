import numpy as np
import scipy
from PIL import Image, ImageDraw, ImageFont


# This code allows gifs to be saved of the training episode for use in the Control Center.
def save_gif(frames, frame_shape, episode, rl='dqn'):
    height, width = frame_shape[0], frame_shape[1]
    images = np.reshape(np.array(frames), [len(frames), height, width])
    images_ori = images[:]
    if images.shape[1] != 3:
        images = color_frame_continuous(images)
    big_images = []
    for image in images:
        big_images.append(scipy.misc.imresize(image, [height * 30, width * 30], interp='nearest'))
    big_images = np.array(big_images)
    make_gif(big_images, images_ori, '../frames/%s/%d-%d/image' % (rl, width, height) + str(episode) + '.gif')


def make_gif(images, images_ori, fname):
    imgs = []
    fnt = ImageFont.truetype("arial.ttf", 10)
    for k in range(len(images)):
        base = Image.fromarray(np.uint8(images[k])).convert('RGBA')
        txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        for i in range(int(base.size[1]/30)):
            for j in range(int(base.size[0]/30)):
                if images_ori[k][i][j] > 0:
                    d.text((j*30+12, i*30+12), "{0}".format(images_ori[k][i][j]), font=fnt, fill=(255, 255, 255, 255))
        img = Image.alpha_composite(base, txt)
        imgs.append(img)
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], optimize=False, duration=400, loop=0)


def color_frame_continuous(images, dim=2):
    if dim == 2:
        colored_images = np.zeros([len(images), images.shape[1], images.shape[2], 3])
        for k in range(len(images)):
            for i in range(images.shape[1]):
                for j in range(images.shape[2]):
                    if images[k, i, j] == -1.0:
                        colored_images[k, i, j] = [0, 0, 0]
                    else:
                        colored_images[k, i, j] = [0, min(255, images[k, i, j]), 0]
    return colored_images