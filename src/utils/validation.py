import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.losses.utils import bbox_oiou, bbox_overlaps


def _recalls_sim(sims, labels, thrs):
    recalls = np.zeros(thrs.size)
    for i, thr in enumerate(thrs):
        tn, fp, fn, tp = confusion_matrix(sims >= thr, labels).ravel()
        recalls[i] = tp / (tp + fn)
    return recalls


def _recalls(ious, thrs):
    img_num = ious.shape[0]
    recalls = np.zeros(thrs.size)
    for i, thr in enumerate(thrs):
        recalls[i] = (ious >= thr).sum() / float(img_num)
    return recalls


def eval_recalls(ious, iou_thrs=0.5, logger=None, label=None):
    """Calculate recalls.

    Args:
        ious (list[ndarray]): a list of arrays of shape (n, 4)
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    if label is not None:
        recalls = _recalls_sim(np.array(ious), np.array(label), np.array(iou_thrs))
    else:
        recalls = _recalls(np.array(ious), np.array(iou_thrs))
    if logger:
        logger.info('Recalls\t R0.5\t R0.75\t R0.9\t')
        logger.info('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(recalls[0], recalls[5], recalls[8]))
    else:
        print('Recalls\t R0.5\t R0.75\t R0.9\t')
        print('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(recalls[0], recalls[5], recalls[8]))
    return recalls


@torch.no_grad()
def evaluate(model,
             dataloader,
             logger,
             iou_thrs=np.arange(0.5, 0.96, 0.05),
             oiou=False,
             save_results=True,
             output_dir='outputs'):
    ious = []
    oious = []
    sims = []
    labels = []

    # Prepare list to save results
    results_records = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # data = model(batch)

        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].cuda()
        data = model(batch)
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].cpu()

        bs = batch['overlap_box1'].shape[0]
        label = torch.cat((torch.ones(bs), torch.zeros(bs))).to(torch.int64)
        sim = [(k[0] / k.sum()).item() for k in data['sim'].cpu()]
        sims += sim
        labels += label

        if not oiou:
            ious1 = bbox_overlaps(batch['overlap_box1'], data['pred_bbox1'][:bs].cpu(), is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'], data['pred_bbox2'][:bs].cpu(), is_aligned=True)

            # Save results for each image pair
            if save_results:
                for j in range(bs):
                    # Extract paths from batch
                    if isinstance(batch['image_path1'], (list, tuple)):
                        im1_path = batch['image_path1'][j]
                        im2_path = batch['image_path2'][j]
                    else:
                        # If not list, batch size is 1
                        im1_path = batch['image_path1']
                        im2_path = batch['image_path2']

                    avg_iou = (ious1[j].item() + ious2[j].item()) / 2
                    results_records.append(f"{im1_path} {im2_path} {avg_iou:.6f} 0.000000")
        else:
            ious1 = bbox_overlaps(batch['overlap_box1'], data['pred_bbox1'][:bs].cpu(), is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'], data['pred_bbox2'][:bs].cpu(), is_aligned=True)
            oious1 = bbox_oiou(batch['overlap_box1'], data['pred_bbox1'][:bs].cpu())
            oious2 = bbox_oiou(batch['overlap_box2'], data['pred_bbox2'][:bs].cpu())
            oious += list(oious1.numpy()) + list(oious2.numpy())

            # Save results for each image pair
            if save_results:
                for j in range(bs):
                    # Extract paths from batch
                    if isinstance(batch['image_path1'], (list, tuple)):
                        im1_path = batch['image_path1'][j]
                        im2_path = batch['image_path2'][j]
                    else:
                        # If not list, batch size is 1
                        im1_path = batch['image_path1']
                        im2_path = batch['image_path2']

                    avg_iou = (ious1[j].item() + ious2[j].item()) / 2
                    avg_oiou = (oious1[j].item() + oious2[j].item()) / 2
                    results_records.append(f"{im1_path} {im2_path} {avg_iou:.6f} {avg_oiou:.6f}")

        ious += list(ious1.numpy()) + list(ious2.numpy())

    logger.info('mIoU')
    logger.info('Values\t {:.5f}\t'.format(np.mean(ious)))
    logger.info('SIMILARITY')
    eval_recalls(sims, iou_thrs, logger, labels)
    logger.info('IoU')
    eval_recalls(ious, iou_thrs, logger)
    if oiou:
        logger.info('OIoU')
        eval_recalls(oious, iou_thrs, logger)

    # Save detailed results to txt
    if save_results and results_records:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'evaluation_results.txt')

        with open(results_file, 'w') as f:
            # Write header
            if oiou:
                f.write("# im1path im2path avg_iou avg_oiou\n")
            else:
                f.write("# im1path im2path avg_iou avg_oiou(not_computed)\n")

            # Write all results
            for record in results_records:
                f.write(record + '\n')

        logger.info(f'Detailed results saved to: {results_file}')
        logger.info(f'Total records saved: {len(results_records)}')
