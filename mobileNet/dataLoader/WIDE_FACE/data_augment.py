import cv2
import numpy as np
import random
# from box_utils import matrix_iof
minLen = 32.*0.9
import uuid
import os


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > minLen
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def boxTooSmall(box, r, threshold):
    if (box[3] - box[1])*r < threshold and (box[2] - box[0])*r < threshold:
        return True
    return False


def boxInArea(area, box):
    borderThresh = 0.3 # 边缘超出区域的阈值
    areaThresh = 0.4 # 面积阈值
    l, t, r, d = box
    w, h = r - l, d - t
    tl, tt, tr, td = box
    if box[0] < area[0]:
        tl = area[0]
    if box[1] < area[1]:
        tt = area[1]
    if box[2] > area[2]:
        tr = area[2]
    if box[3] > area[3]:
        td = area[3]

    tw, th = tr - tl, td - tt
    if tw <= 0 or th <= 0: # 不重合
        return False, None
    else:
        if (tt - t)/h < borderThresh and (d - td) / h < borderThresh and (tt - t + d - td)/h < borderThresh and \
                (tl - l)/w < borderThresh and (r - tr)/w < borderThresh and (tl - l + r - tr)/w < borderThresh and \
                tw*th/(w*h) > areaThresh:
            return True, np.array([tl - area[0], tt - area[1], tr - area[0], td - area[1]])
        else:
            return False, None


def mosaic(selected_image, nsize=3):
    rows, cols = selected_image.shape
    dist = selected_image.copy()
    # 划分小方块，每个小方块填充随机颜色
    for y in range(0, rows, nsize):
        for x in range(0, cols, nsize):
            dist[y:y+nsize, x:x+nsize] = (np.random.randint(0, 255))
    return dist


def fillDelArea(image, dBoxes):
    for box in dBoxes:
        image_t = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        ave = image_t.mean()
        tmp = np.ones_like(image_t, dtype=np.float32)*ave
        image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = tmp


def _cropFace(image, boxes, labels, img_dim):
    h, w, _ = image.shape
    pad_image_flag = False
    for i in range(150):
        # 人脸标签范围扩展
        if random.uniform(0, 1) <= 0.2:
            scalew = 1
            scaleh = 1
        else:
            scalew = random.uniform(0.2, 1.)
            scaleh = random.uniform(0.2, 1.)
        tw = int(scalew * w)
        th = int(scaleh * h)

        # 过小
        if (tw < img_dim//4 or th < img_dim//4) and th/tw < 3: # 不会过小，且长宽比合适
            continue
        r = img_dim / float(tw)

        if tw == w:
            l = 0
        else:
            l = random.randint(0, w - tw - 1)
        if th == h:
            t = 0
        else:
            t = random.randint(0, h - th - 1)

        roi = (l, t, l+tw, t+th)

        # check box
        tmpBoxes = []
        tmpLabels = []
        delBoxes = []
        # 是不是在区域中，是不是过小
        for k in range(len(boxes)):
            box = boxes[k]
            flag, newBox = boxInArea(roi, box)
            if flag and not boxTooSmall(newBox, r, minLen):
                tmpBoxes.append(newBox*r)
                tmpLabels.append(labels[k])
            elif flag:
                delBoxes.append(newBox*r)
            else:
                continue

        if not tmpBoxes:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        image_t = cv2.resize(image_t, (0, 0), fx=r, fy=r)

        if delBoxes:
            fillDelArea(image_t, delBoxes)
        pad_image_flag = False

        return image_t, np.array(tmpBoxes), np.array(tmpLabels), pad_image_flag
    return image, boxes, labels, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean, rgb_norm):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    h, w, _ = image.shape
    r = insize/w
    image = cv2.resize(image, (0, 0), fx=r, fy=r, interpolation=interp_method)
    image = image.astype(np.float32)
    # image -= rgb_mean
    # image *= rgb_norm
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def drawBoxes(image, boxes, storePath):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        cv2.putText(image, str(int(box[2] - box[0])), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(storePath, image)


class preproc(object):

    def __init__(self, img_dim, rgb_means, rgb_norm, drawLable=False):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.rgb_norm = rgb_norm
        self.draw = drawLable

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        #image_t = _distort(image)
        #image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        #image_t, boxes_t, labels_t = _crop(image_t, boxes, labels, self.img_dim, self.rgb_means)
        # image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t, boxes_t, labels_t, pad_image_flag = _cropFace(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t = _mirror(image_t, boxes_t)
        height, width, _ = image_t.shape

        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means, self.rgb_norm)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        delInds = np.where(np.logical_or((boxes_t[:, 2] - boxes_t[:, 0])*self.img_dim < minLen, (boxes_t[:, 3] - boxes_t[:, 1])*self.img_dim < minLen))
        tmpDI = np.squeeze(delInds)
        if tmpDI.size > 0:
            delBoxes = boxes_t[tmpDI]
        boxes_t = np.delete(boxes_t, np.squeeze(delInds), axis=0)
        labels_t = np.delete(labels_t, np.squeeze(delInds), axis=0)

        if boxes_t.size < 1:
            return None, None

        if tmpDI.size > 0:
            delBoxes *= self.img_dim
            fillDelArea(image_t, delBoxes if tmpDI.size > 1 else [delBoxes])

        if self.draw:
            storePath = './tmpImgs'
            if not os.path.exists(storePath):
                os.mkdir(storePath)
            num = len(os.listdir(storePath))
            if num < 12600: # 数量控制
                img = image_t
                drawBoxes(img.astype(np.int32), boxes_t*self.img_dim, os.path.join(storePath, str(uuid.uuid4()) + '.jpg'))

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t
