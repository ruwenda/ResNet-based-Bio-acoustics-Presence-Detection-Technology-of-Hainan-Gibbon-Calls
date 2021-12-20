import os
import torch
from utils.Data_Loader import getDataLoader
from model.BPDnet import BPDnet
from catalyst import dl
from adamp import AdamP
from timm.loss import LabelSmoothingCrossEntropy


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    # parameter c select：
    num_channel = [8, 16, 32, 64]

    for channel in num_channel:
        model = BPDnet(channel, isSpecAugmentation=True)
        logDir = f"./log/BPDnet{channel}_specAugment"
        num_epochs = 60

        # 训练参数
        batchSize = 128
        lr = 3e-4

        criterion = LabelSmoothingCrossEntropy()
        trainDataLoader = getDataLoader(r"Data/Augmented_Image_Data", batchSize=batchSize)
        validDataloader = getDataLoader(r"Data/VAL_Augmented_Image_Data", batchSize=batchSize)
        loaders = {
            "train": trainDataLoader,
            "valid": validDataloader,
        }

        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )

        # model training
        optimizer = AdamP(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                        steps_per_epoch=len(trainDataLoader),
                                                        epochs=num_epochs, verbose=False)
        runner.train(
            model=model,
            scheduler=scheduler,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=num_epochs,
            callbacks=[
                dl.CheckpointCallback(logdir=os.path.join(logDir, "Model"), loader_key="valid",
                                      metric_key="loss", minimize=True),
                dl.OptimizerCallback(metric_key="loss", accumulation_steps=1,
                                     grad_clip_fn=torch.nn.utils.clip_grad_norm_,
                                     grad_clip_params={"max_norm": 2.0, "norm_type": 2.0},),
                dl.SchedulerCallback(loader_key="train", metric_key="_batch_", mode="batch"),
            ],
            loggers={"tensorboard": dl.TensorboardLogger(logdir=os.path.join(logDir, "tensorboard"))},
            logdir=logDir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=False,
        )
