from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ReweightHook(Hook):

    def __init__(self, step):
        self.step = step

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        bbox_head = runner.model.module.roi_head.bbox_head
        epoch = getattr(bbox_head, '_epoch')
        epoch += 1
        setattr(bbox_head, '_epoch', epoch)
        if epoch == self.step:
            loss_cls = runner.model.module.roi_head.bbox_head.loss_cls
            setattr(loss_cls, 'reweight', True)
            runner.logger.info("Reweight of loss_cls: {}".format(loss_cls))
        start_epoch = getattr(bbox_head, 'start_epoch')
        if epoch == start_epoch + 1:
            setattr(bbox_head, 'flag', True)
            runner.logger.info("FHM begins working")

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
