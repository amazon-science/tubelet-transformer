class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val, n=1):
        if n>0:
            self.val = val
            if self.count == 0:
                self.avg = self.val
            else:
                self.avg = self.avg*self.momentum + (1-self.momentum)* val
            self.count += n
