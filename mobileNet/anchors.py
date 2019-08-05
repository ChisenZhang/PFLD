# !/usr/bin/python
# encoding:utf-8

'''
anchor 生成
'''

import numpy as np


# anchor 两点转中心点加长宽
def anchorPointTrans(anchors):
    w, h = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    anchors[:, 0] = anchors[:, 0] + w/2.
    anchors[:, 1] = anchors[:, 1] + h/2.
    anchors[:, 2] = w
    anchors[:, 3] = h
    return anchors


class Anchors(object):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [4, 5]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** x for x in self.pyramid_levels]
        if ratios is None:
            # self.ratios = np.array([1., 1.5, 2., 2.5, 3.])
            self.ratios = np.array([1., 1.5])
        if scales is None:
            self.scales = np.array([1, 2, 4, 8])

    # def forward(self, image):
    #
    #     image_shape = image.shape[2:]
    #     image_shape = np.array(image_shape)
    #     image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
    #
    #     # compute anchors over all pyramid levels
    #     all_anchors = np.zeros((0, 4)).astype(np.float32)
    #
    #     for idx, p in enumerate(self.pyramid_levels):
    #         anchors = self.generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
    #         shifted_anchors = self.shift(image_shapes[idx], self.strides[idx], anchors)
    #         all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    #
    #     all_anchors = np.expand_dims(all_anchors, axis=0)
    #
    #     return all_anchors

    def get_anchors(self, fmSizes, ratios=np.array([1., 1.5]), scales=np.array([[2], [8, 16, 32]]), strides=[5, 6],
                    fmBased=True, imgSize=256.):
        anchorsAll = np.zeros((0, 4))

        if fmBased:
            if len(fmSizes) != ratios.size or len(fmSizes) != scales.size:
                print('getAnchors: fmSizes not equ ratios or scales len')
                exit(-1)

            for i in range(len(fmSizes)):
                fmSize = fmSizes[i]
                tmpAnchors = self.generate_anchors(fmSize, ratios, scales[i])
                sifted_anchors = self.shift(fmSize, 2**strides[i], anchors=tmpAnchors)
                anchorsAll = np.append(anchorsAll, sifted_anchors, axis=0)
        else:
            if scales.size != len(strides):
                print('getAnchors tile: scale and stride len not equ')
                exit(-1)
            for i in range(len(strides)):
                tmpAnchors = self.generate_anchors_tile(ratios, scales[i]*fmSizes[i][0])
                sifted_anchors = self.sift_tile(tmpAnchors, stride=2**strides[i])
                anchorsAll = np.append(anchorsAll, sifted_anchors, axis=0)

        return anchorsAll/imgSize

    def generate_anchors_tile(self, ratios=None, scales=None):
        if ratios is None:
            ratios = np.array([1., 1.5])

        if scales is None:
            scales = np.array([32, 64, 128, 256])

        # initialize output anchors
        anchors = np.zeros((ratios.size*len(scales), 4))

        # scale base_size
        anchors[:, 2:] = np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def sift_tile(self, anchors, imgSize=[256, 256], stride=16):
        step = 2
        shift_x = (np.arange(0, imgSize[1]-(step - 1)*stride/step, stride/step))
        shift_y = (np.arange(0, imgSize[0]-(step - 1)*stride/step, stride/step))
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors


    def generate_anchors(self, base_size=32, ratios=None, scales=None):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """

        if ratios is None:
            ratios = np.array([1., 1.5])

        if scales is None:
            scales = np.array([1, 2, 4, 8])

        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors


    def compute_shape(self, image_shape, pyramid_levels):
        """Compute shapes based on pyramid levels.

        :param image_shape:
        :param pyramid_levels:
        :return:
        """
        image_shape = np.array(image_shape[:2])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        return image_shapes


    def anchors_for_shape(self,
            image_shape,
            pyramid_levels=None,
            ratios=None,
            scales=None,
            strides=None,
            sizes=None,
            shapes_callback=None,
    ):
        image_shapes = self.compute_shape(image_shape, pyramid_levels)

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4))
        for idx, p in enumerate(pyramid_levels):
            anchors = self.generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
            shifted_anchors = self.shift(image_shapes[idx], strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return all_anchors


    def shift(self, shape, stride, anchors):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors

if __name__ == '__main__':
    anchorsC = Anchors()
    anchors = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True)
    print('abc')
