# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.modeling.discriminator import FCOSDiscriminator, FCOSDiscriminator_CA
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir



def train(cfg, local_rank, distributed):
    ##########################################################################
    ############################# Initial Model ##############################
    ##########################################################################
    model = {}
    device = torch.device(cfg.MODEL.DEVICE)

    backbone = build_backbone(cfg).to(device)
    fcos = build_rpn(cfg, backbone.out_channels).to(device)

    if cfg.MODEL.ADV.USE_DIS_GLOBAL:
        if cfg.MODEL.ADV.USE_DIS_P7:
            dis_P7 = FCOSDiscriminator(
                num_convs=cfg.MODEL.ADV.DIS_P7_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P7,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P6:
            dis_P6 = FCOSDiscriminator(
                num_convs=cfg.MODEL.ADV.DIS_P6_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P6,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P5:
            dis_P5 = FCOSDiscriminator(
                num_convs=cfg.MODEL.ADV.DIS_P5_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P5,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P4:
            dis_P4 = FCOSDiscriminator(
                num_convs=cfg.MODEL.ADV.DIS_P4_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P4,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P3:
            dis_P3 = FCOSDiscriminator(
                num_convs=cfg.MODEL.ADV.DIS_P3_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P3,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)

    if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
        if cfg.MODEL.ADV.USE_DIS_P7:
            dis_P7_CA = FCOSDiscriminator_CA(
                num_convs=cfg.MODEL.ADV.CA_DIS_P7_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P7,
                center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P6:
            dis_P6_CA = FCOSDiscriminator_CA(
                num_convs=cfg.MODEL.ADV.CA_DIS_P6_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P6,
                center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P5:
            dis_P5_CA = FCOSDiscriminator_CA(
                num_convs=cfg.MODEL.ADV.CA_DIS_P5_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P5,
                center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P4:
            dis_P4_CA = FCOSDiscriminator_CA(
                num_convs=cfg.MODEL.ADV.CA_DIS_P4_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P4,
                center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_P3:
            dis_P3_CA = FCOSDiscriminator_CA(
                num_convs=cfg.MODEL.ADV.CA_DIS_P3_NUM_CONVS,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P3,
                center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        fcos = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fcos)

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3)

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7_CA)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6_CA)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5_CA)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4_CA)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3_CA)

    ##########################################################################
    #################### Initial Optimizer and Scheduler #####################
    ##########################################################################
    optimizer = {}
    optimizer["backbone"] = make_optimizer(cfg, backbone, name='backbone')
    optimizer["fcos"] = make_optimizer(cfg, fcos, name='fcos')

    if cfg.MODEL.ADV.USE_DIS_GLOBAL:
        if cfg.MODEL.ADV.USE_DIS_P7:
            optimizer["dis_P7"] = make_optimizer(cfg, dis_P7, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P6:
            optimizer["dis_P6"] = make_optimizer(cfg, dis_P6, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P5:
            optimizer["dis_P5"] = make_optimizer(cfg, dis_P5, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P4:
            optimizer["dis_P4"] = make_optimizer(cfg, dis_P4, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P3:
            optimizer["dis_P3"] = make_optimizer(cfg, dis_P3, name='discriminator')

    if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
        if cfg.MODEL.ADV.USE_DIS_P7:
            optimizer["dis_P7_CA"] = make_optimizer(cfg, dis_P7_CA, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P6:
            optimizer["dis_P6_CA"] = make_optimizer(cfg, dis_P6_CA, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P5:
            optimizer["dis_P5_CA"] = make_optimizer(cfg, dis_P5_CA, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P4:
            optimizer["dis_P4_CA"] = make_optimizer(cfg, dis_P4_CA, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P3:
            optimizer["dis_P3_CA"] = make_optimizer(cfg, dis_P3_CA, name='discriminator')

    scheduler = {}
    scheduler["backbone"] = make_lr_scheduler(cfg, optimizer["backbone"], name='backbone')
    scheduler["fcos"] = make_lr_scheduler(cfg, optimizer["fcos"], name='fcos')

    if cfg.MODEL.ADV.USE_DIS_GLOBAL:
        if cfg.MODEL.ADV.USE_DIS_P7:
            scheduler["dis_P7"] = make_lr_scheduler(cfg, optimizer["dis_P7"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P6:
            scheduler["dis_P6"] = make_lr_scheduler(cfg, optimizer["dis_P6"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P5:
            scheduler["dis_P5"] = make_lr_scheduler(cfg, optimizer["dis_P5"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P4:
            scheduler["dis_P4"] = make_lr_scheduler(cfg, optimizer["dis_P4"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P3:
            scheduler["dis_P3"] = make_lr_scheduler(cfg, optimizer["dis_P3"], name='discriminator')

    if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
        if cfg.MODEL.ADV.USE_DIS_P7:
            scheduler["dis_P7_CA"] = make_lr_scheduler(cfg, optimizer["dis_P7_CA"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P6:
            scheduler["dis_P6_CA"] = make_lr_scheduler(cfg, optimizer["dis_P6_CA"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P5:
            scheduler["dis_P5_CA"] = make_lr_scheduler(cfg, optimizer["dis_P5_CA"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P4:
            scheduler["dis_P4_CA"] = make_lr_scheduler(cfg, optimizer["dis_P4_CA"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_P3:
            scheduler["dis_P3_CA"] = make_lr_scheduler(cfg, optimizer["dis_P3_CA"], name='discriminator')

    ##########################################################################
    ######################## DistributedDataParallel #########################
    ##########################################################################
    if distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )
        fcos = torch.nn.parallel.DistributedDataParallel(
            fcos, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.parallel.DistributedDataParallel(
                    dis_P7, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.parallel.DistributedDataParallel(
                    dis_P6, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.parallel.DistributedDataParallel(
                    dis_P5, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.parallel.DistributedDataParallel(
                    dis_P4, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.parallel.DistributedDataParallel(
                    dis_P3, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P7_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P6_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P5_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P4_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P3_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

    ##########################################################################
    ########################### Save Model to Dict ###########################
    ##########################################################################
    model["backbone"] = backbone
    model["fcos"] = fcos

    if cfg.MODEL.ADV.USE_DIS_GLOBAL:
        if cfg.MODEL.ADV.USE_DIS_P7:
            model["dis_P7"] = dis_P7
        if cfg.MODEL.ADV.USE_DIS_P6:
            model["dis_P6"] = dis_P6
        if cfg.MODEL.ADV.USE_DIS_P5:
            model["dis_P5"] = dis_P5
        if cfg.MODEL.ADV.USE_DIS_P4:
            model["dis_P4"] = dis_P4
        if cfg.MODEL.ADV.USE_DIS_P3:
            model["dis_P3"] = dis_P3

    if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
        if cfg.MODEL.ADV.USE_DIS_P7:
            model["dis_P7_CA"] = dis_P7_CA
        if cfg.MODEL.ADV.USE_DIS_P6:
            model["dis_P6_CA"] = dis_P6_CA
        if cfg.MODEL.ADV.USE_DIS_P5:
            model["dis_P5_CA"] = dis_P5_CA
        if cfg.MODEL.ADV.USE_DIS_P4:
            model["dis_P4_CA"] = dis_P4_CA
        if cfg.MODEL.ADV.USE_DIS_P3:
            model["dis_P3_CA"] = dis_P3_CA

    ##########################################################################
    ################################ Training ################################
    ##########################################################################
    arguments = {}
    arguments["iteration"] = 0
    arguments["use_dis_global"] = cfg.MODEL.ADV.USE_DIS_GLOBAL
    arguments["use_dis_ca"] = cfg.MODEL.ADV.USE_DIS_CENTER_AWARE
    arguments["ga_dis_lambda"] = cfg.MODEL.ADV.GA_DIS_LAMBDA
    arguments["ca_dis_lambda"] = cfg.MODEL.ADV.CA_DIS_LAMBDA

    arguments["use_feature_layers"] = []
    if cfg.MODEL.ADV.USE_DIS_P7:
        arguments["use_feature_layers"].append("P7")
    if cfg.MODEL.ADV.USE_DIS_P6:
        arguments["use_feature_layers"].append("P6")
    if cfg.MODEL.ADV.USE_DIS_P5:
        arguments["use_feature_layers"].append("P5")
    if cfg.MODEL.ADV.USE_DIS_P4:
        arguments["use_feature_layers"].append("P4")
    if cfg.MODEL.ADV.USE_DIS_P3:
        arguments["use_feature_layers"].append("P3")

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, load_dis=True, load_opt_sch=False)
    # arguments.update(extra_checkpoint_data)

    # Initial dataloader (both target and source domain)
    data_loader = {}
    data_loader["source"] = make_data_loader_source(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    data_loader["target"] = make_data_loader_target(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model["backbone"] = model["backbone"].module
        model["fcos"] = model["fcos"].module
        #if cfg.MODEL.ADV.USE_DIS_P7:
        #    model["dis_P7"] = model["dis_P7"].module
        #if cfg.MODEL.ADV.USE_DIS_P6:
        #    model["dis_P6"] = model["dis_P6"].module
        #if cfg.MODEL.ADV.USE_DIS_P5:
        #    model["dis_P5"] = model["dis_P5"].module
        #if cfg.MODEL.ADV.USE_DIS_P4:
        #    model["dis_P4"] = model["dis_P4"].module
        #if cfg.MODEL.ADV.USE_DIS_P3:
        #    model["dis_P3"] = model["dis_P3"].module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Check if domain adaption
    assert cfg.MODEL.DA_ON, "Domain Adaption"

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    main()
