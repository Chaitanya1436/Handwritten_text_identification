class Batch:
    def __init__(self, imgs, gt_texts, size):
        self.imgs = imgs
        self.gt_texts = gt_texts
        self.batch_size = size
