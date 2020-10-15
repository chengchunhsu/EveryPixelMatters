# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list


def foward_detector(model, images, targets=None, return_maps=False):
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    images = to_image_list(images)
    features = model_backbone(images.tensors)

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    losses = {}

    if model_fcos.training and targets is None:
        # train G on target domain
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=None, return_maps=return_maps)
        assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0  # loss_dict should be empty dict
    else:
        # train G on source domain / inference
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

    if model_fcos.training:
        # training
        m = {
            layer: {
                map_type:
                score_maps[map_type][map_layer_to_index[layer]]
                for map_type in score_maps
            }
            for layer in feature_layers
        }
        losses.update(proposal_losses)
        return losses, f, m
    else:
        # inference
        result = proposals
        return result


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    USE_DIS_GLOBAL = arguments["use_dis_global"]
    USE_DIS_CENTER_AWARE = arguments["use_dis_ca"]
    used_feature_layers = arguments["use_feature_layers"]

    # dataloader
    data_loader_source = data_loader["source"]
    data_loader_target = data_loader["target"]

    # classified label of source domain and target domain
    source_label = 1.0
    target_label = 0.0

    # dis_lambda
    if USE_DIS_GLOBAL:
        ga_dis_lambda = arguments["ga_dis_lambda"]
    if USE_DIS_CENTER_AWARE:
        ca_dis_lambda = arguments["ca_dis_lambda"]

    # Start training
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")

    # model.train()
    for k in model:
        model[k].train()

    meters = MetricLogger(delimiter="  ")
    assert len(data_loader_source) == len(data_loader_target)
    max_iter = max(len(data_loader_source), len(data_loader_target))
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, ((images_s, targets_s, _), (images_t, _, _)) \
        in enumerate(zip(data_loader_source, data_loader_target), start_iter):

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            # scheduler.step()
            for k in scheduler:
                scheduler[k].step()

        images_s = images_s.to(device)
        targets_s = [target_s.to(device) for target_s in targets_s]
        images_t = images_t.to(device)
        # targets_t = [target_t.to(device) for target_t in targets_t]

        # optimizer.zero_grad()
        for k in optimizer:
            optimizer[k].zero_grad()

        ##########################################################################
        #################### (1): train G with source domain #####################
        ##########################################################################

        loss_dict, features_s, score_maps_s = foward_detector(
            model, images_s, targets=targets_s, return_maps=True)

        # rename loss to indicate domain
        loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

        losses.backward(retain_graph=True)
        del loss_dict, losses

        ##########################################################################
        #################### (2): train D with source domain #####################
        ##########################################################################

        loss_dict = {}
        for layer in used_feature_layers:
            # detatch score_map
            for map_type in score_maps_s[layer]:
                score_maps_s[layer][map_type] = score_maps_s[layer][map_type].detach()
            if USE_DIS_GLOBAL:
                loss_dict["loss_adv_%s_ds" % layer] = \
                    ga_dis_lambda * model["dis_%s" % layer](features_s[layer], source_label, domain='source')
            if USE_DIS_CENTER_AWARE:
                loss_dict["loss_adv_%s_CA_ds" % layer] = \
                    ca_dis_lambda * model["dis_%s_CA" % layer](features_s[layer], source_label, score_maps_s[layer], domain='source')

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_ds=losses_reduced, **loss_dict_reduced)

        losses.backward()
        del loss_dict, losses

        ##########################################################################
        #################### (3): train D with target domain #####################
        #################################################################

        loss_dict, features_t, score_maps_t = foward_detector(model, images_t, return_maps=True)
        assert len(loss_dict) == 1 and loss_dict["zero"] == 0  # loss_dict should be empty dict

        # loss_dict["loss_adv_Pn"] = model_dis_Pn(features_t["Pn"], target_label, domain='target')
        for layer in used_feature_layers:
            # detatch score_map
            for map_type in score_maps_t[layer]:
                score_maps_t[layer][map_type] = score_maps_t[layer][map_type].detach()
            if USE_DIS_GLOBAL:
                loss_dict["loss_adv_%s_dt" % layer] = \
                    ga_dis_lambda * model["dis_%s" % layer](features_t[layer], target_label, domain='target')
            if USE_DIS_CENTER_AWARE:
                loss_dict["loss_adv_%s_CA_dt" %layer] = \
                    ca_dis_lambda * model["dis_%s_CA" % layer](features_t[layer], target_label, score_maps_t[layer], domain='target')

        losses = sum(loss for loss in loss_dict.values())

        # del "zero" (useless after backward)
        del loss_dict['zero']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss_dt=losses_reduced, **loss_dict_reduced)

        # saved GRL gradient
        grad_list = []
        for layer in used_feature_layers:
            def save_grl_grad(grad):
                grad_list.append(grad)
            features_t[layer].register_hook(save_grl_grad)

        losses.backward()

        # Uncomment to log GRL gradient
        grl_grad = {}
        grl_grad_log = {}
        # grl_grad = {
        #     layer: grad_list[i]
        #     for i, layer in enumerate(used_feature_layers)
        # }
        # for layer in used_feature_layers:
        #     saved_grad = grl_grad[layer]
        #     grl_grad_log["grl_%s_abs_mean" % layer] = torch.mean(
        #         torch.abs(saved_grad)) * 10e4
        #     grl_grad_log["grl_%s_mean" % layer] = torch.mean(saved_grad) * 10e6
        #     grl_grad_log["grl_%s_std" % layer] = torch.std(saved_grad) * 10e6
        #     grl_grad_log["grl_%s_max" % layer] = torch.max(saved_grad) * 10e6
        #     grl_grad_log["grl_%s_min" % layer] = torch.min(saved_grad) * 10e6
        # meters.update(**grl_grad_log)

        del loss_dict, losses, grad_list, grl_grad, grl_grad_log

        ##########################################################################
        ##########################################################################
        ##########################################################################

        # optimizer.step()
        for k in optimizer:
            optimizer[k].step()

        if pytorch_1_1_0_or_later:
            # scheduler.step()
            for k in scheduler:
                scheduler[k].step()

        # End of training
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        sample_layer = used_feature_layers[0]  # sample any one of used feature layer
        if USE_DIS_GLOBAL:
            sample_optimizer = optimizer["dis_%s" % sample_layer]
        if USE_DIS_CENTER_AWARE:
            sample_optimizer = optimizer["dis_%s_CA" % sample_layer]
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join([
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr_backbone: {lr_backbone:.6f}",
                    "lr_fcos: {lr_fcos:.6f}",
                    "lr_dis: {lr_dis:.6f}",
                    "max mem: {memory:.0f}",
                ]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                    lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                    lr_dis=sample_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ))
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        # if iteration > 5000 and iteration <= 6000 and iteration % 50 == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))
