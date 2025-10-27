import sys
from pathlib import Path

import argparse
import cv2
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.nn
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from typing import List

from DataUtils import FakePartsV2DatasetBase
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from core.utils1.utils import get_network, str2bool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    datefmt='%m/%d/%Y %I:%M:%S'
)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))


def args_parse():
    parser = argparse.ArgumentParser()

    # RAFT / optical flow backbone
    parser.add_argument('--raft_model', type=str, required=True,
                        help="Path to RAFT checkpoint (optical flow backbone).")
    # RAFT forward config flags expected by RAFT()
    parser.add_argument('--small', action='store_true', help="RAFT small model flag")
    parser.add_argument('--mixed_precision', action='store_true', help="Use mixed precision in RAFT")
    parser.add_argument('--alternate_corr', action='store_true', help="Use alternate correlation in RAFT")

    # RGB / OF classifiers
    parser.add_argument("--model_original_path", type=str, required=True, help="Checkpoint for RGB classifier.")
    parser.add_argument("--model_optical_flow_path", type=str, required=True,
                        help="Checkpoint for OpticalFlow classifier.")
    parser.add_argument("--arch", type=str, default="resnet50")

    # Data / output
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--done_csv_list", nargs="*", default=None)
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--method_name", type=str, default=None,
                        help="If set, only evaluate samples with this method name.")

    # Eval behaviour
    parser.add_argument("--batch_size", type=int, default=1, help="Num videos per batch. 1 is safest for memory.")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on mean combined score.")
    parser.add_argument("--aug_norm", type=str2bool, default=True, help="Apply normalisation before inference.")
    parser.add_argument("--model_name", type=str, default="AIGVDet", help="Name stored in the 'model' column.")

    return parser.parse_args()


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(self, data_root, csv_path, done_csv_list, transform=None, model_name="unknown_model",
                 on_corrupt="warn"):
        super().__init__(
            data_root=data_root,
            mode="video",
            csv_path=csv_path,
            done_csv_list=done_csv_list or [],
            model_name=model_name,
            transform=transform,
            on_corrupt=on_corrupt,
        )

    @property
    def methods(self):
        return self._methods

    def __getitem__(self, idx):
        base_item = super().__getitem__(idx)
        if base_item is None:
            return None

        cap, label, meta = base_item
        video_path = str(self._abs_paths[idx])
        cap.release()

        return video_path, label, meta

    # def __getitem__(self, idx):
    #     base_item = super().__getitem__(idx)
    #     if base_item is None:
    #         return None
    #
    #     cap, label, meta = base_item  # cap is cv2.VideoCapture from base class
    #     frames: List[torch.Tensor] = []
    #     while True:
    #         ok, frame_bgr = cap.read()
    #         if not ok:
    #             break
    #         # BGR -> RGB
    #         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #         pil_im = Image.fromarray(frame_rgb)
    #
    #         if self.transform is not None:
    #             frame_t = self.transform(pil_im)  # float32 [0,1], shape [3,H,W]
    #         else:
    #             frame_t = TF.to_tensor(pil_im)
    #
    #         frames.append(frame_t)
    #
    #     cap.release()
    #
    #     if len(frames) == 0:
    #         # nothing decoded, treat as corrupt sample
    #         if self.on_corrupt == "warn":
    #             print(f"[warn] Video {meta['sample_id']} had 0 frames after decode.")
    #         return None
    #
    #     # Stack -> [T,3,H,W] float32 in [0,1]
    #     video_tensor = torch.stack(frames, dim=0)
    #     return video_tensor, label, meta


def collate_video_batch(batch):
    """
    After refactor, each item is:
        (video_path:str, label:int, meta:dict)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    video_paths, labels, metas = zip(*batch)
    labels = torch.as_tensor(labels, dtype=torch.long)
    metas = list(metas)
    video_paths = list(video_paths)
    return video_paths, labels, metas


# def collate_video_batch(batch):
#     """
#     Collate function for variable-length videos.
#
#     Input batch: list of (video_tensor[T,3,H,W], label:int, meta:dict) OR None
#     Output:
#         vids:  list[Tensor[T_i,3,H,W]]
#         labels: Tensor[B] int64
#         metas: list[dict] length B
#     """
#     batch = [b for b in batch if b is not None]
#     if not batch:
#         return None
#
#     vids, labels, metas = zip(*batch)  # vids: tuple of [T,3,H,W] tensors (different T)
#     labels = torch.as_tensor(labels, dtype=torch.long)
#     metas = list(metas)
#     vids = list(vids)
#     return vids, labels, metas


def batch_optical_flow_gen(model_flow, img1, img2):
    """
    img1, img2: torch.Tensor [1,3,H,W] already on DEVICE
    Returns: np.ndarray [H,W,3] uint8 colour wheel visualisation
    """
    img1 = img1.float()
    img2 = img2.float()

    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)

    # RAFT forward
    flow_low, flow_up = model_flow(img1, img2, iters=20, test_mode=True)
    # flow_up: [1,2,H,W] flow field
    flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,2] float
    flow_rgb = flow_viz.flow_to_image(flow)  # [H,W,3] uint8
    return flow_rgb


def decode_and_prepare_video(
        video_path: str,
        model_flow,
        aug_norm: bool,
        center_crop_size=(448, 448),
        device=DEVICE,
):
    """
    Returns:
        vid_rgb_norm:  torch.Tensor [T,3,H,W] on DEVICE (float32 or float16 later)
        vid_of_norm:   torch.Tensor [T,3,H,W] on DEVICE
    """
    # 1. decode frames from disk
    cap = cv2.VideoCapture(video_path)
    rgb_frames_cpu = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame_rgb)

        # crop
        pil_im = transforms.CenterCrop(center_crop_size)(pil_im)

        # to tensor [3,H,W] float32 in [0,1]
        frame_t = TF.to_tensor(pil_im)  # still on CPU
        rgb_frames_cpu.append(frame_t)
    cap.release()

    if len(rgb_frames_cpu) == 0:
        return None, None  # caller should handle

    vid_frames_cpu = torch.stack(rgb_frames_cpu, dim=0)  # [T,3,H,W] float32 on CPU, [0,1]

    # 2. normalise rgb branch
    if aug_norm:
        vid_rgb_norm = TF.normalize(
            vid_frames_cpu,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        vid_rgb_norm = vid_frames_cpu

    # move once to GPU
    vid_rgb_norm = vid_rgb_norm.to(device, non_blocking=True)

    # 3. build optical flow visualisation sequence (on GPU for RAFT -> back to CPU then again to GPU)
    T, _, H, W = vid_rgb_norm.shape
    of_tensors = []
    for t in range(T):
        if t < T - 1:
            img1 = vid_rgb_norm[t:t + 1]  # [1,3,H,W] on device
            img2 = vid_rgb_norm[t + 1:t + 2]
        else:
            img1 = vid_rgb_norm[t - 1:t]
            img2 = vid_rgb_norm[t:t + 1]

        flow_rgb = batch_optical_flow_gen(model_flow, img1, img2)  # np[H,W,3] uint8
        flow_t = torch.from_numpy(flow_rgb).permute(2, 0, 1).float()  # [3,H,W], CPU
        flow_t = (flow_t / 255.0) * 2.0 - 1.0  # [-1,1]
        of_tensors.append(flow_t)

    of_video_cpu = torch.stack(of_tensors, dim=0)  # [T,3,H,W] CPU float32
    if aug_norm:
        of_video_norm = TF.normalize(
            of_video_cpu,
            mean=[0.5, 0.5, 0.5],
            std=[0.226, 0.226, 0.226],
        )
    else:
        of_video_norm = of_video_cpu

    of_video_norm = of_video_norm.to(device, non_blocking=True)

    return vid_rgb_norm, of_video_norm


def main():
    args = args_parse()
    os.makedirs(os.path.dirname(args.pred_csv), exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    # 1. Load RGB classifier
    model_rgb = get_network(args.arch)
    sd_rgb = torch.load(args.model_original_path, map_location="cpu")
    sd_rgb = sd_rgb["model"] if "model" in sd_rgb else sd_rgb
    model_rgb.load_state_dict(sd_rgb, strict=True)
    model_rgb.eval().to(DEVICE)

    # 2. Load Optical Flow classifier
    model_of = get_network(args.arch)
    sd_of = torch.load(args.model_optical_flow_path, map_location="cpu")
    sd_of = sd_of["model"] if "model" in sd_of else sd_of
    model_of.load_state_dict(sd_of, strict=True)
    model_of.eval().to(DEVICE)

    # 3. Load RAFT (optical flow extractor)
    model_flow = torch.nn.DataParallel(RAFT(args))
    model_flow.load_state_dict(torch.load(args.raft_model, map_location=torch.device(DEVICE)))
    model_flow.to(DEVICE).eval()

    # 4. Build dataset / loader
    trans_list = [
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),  # -> float32 [0,1]
    ]
    trans = transforms.Compose(trans_list)

    dataset = FakePartsV2Dataset(
        data_root=args.data_root,
        csv_path=args.data_csv,
        done_csv_list=args.done_csv_list or [],
        transform=trans,
        model_name=args.model_name,
        on_corrupt="warn",
    )

    if getattr(args, "method_name", None) is not None:
        filtered_indices = [
            i for i in range(len(dataset))
            if dataset.methods[i] == args.method_name
        ]
        dataset = torch.utils.data.Subset(dataset, filtered_indices)

    # torchrun DDP setup if needed
    sub_csv_path = None
    if world_size > 1:
        all_indices = list(range(len(dataset)))
        shard_indices = [i for i in all_indices if (i % world_size) == rank]
        dataset = torch.utils.data.Subset(dataset, shard_indices)
        # sub temp csv
        if args.data_csv.endswith('.csv'):
            sub_csv_path = f"{args.data_csv[:-4]}_rank{rank}.csv"
        else:
            sub_csv_path = f"{args.data_csv}_rank{rank}.csv"
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_video_batch,
    )

    # 5. Inference loop (video-by-video)
    os.makedirs(os.path.dirname(args.pred_csv), exist_ok=True)
    if sub_csv_path is not None:
        os.makedirs(os.path.dirname(sub_csv_path), exist_ok=True)
    fail_count, fail_threshold = 0, len(loader) // 5 + 1
    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True, desc="Evaluating videos"):
            if batch is None:
                continue
            video_paths_batch, labels_batch, metas_batch = batch
            # video_paths_batch: list[str], length = B (small, e.g. 1 or 2)
            # labels_batch: Tensor[B]
            # metas_batch: list[dict]

            # ---- decode+prepare each video, accumulate for joint forward ----
            all_rgb = []  # list of [T_i,3,H,W] on DEVICE
            all_of = []  # list of [T_i,3,H,W] on DEVICE
            spans = []  # [(start_idx, end_idx, label, meta)]
            cursor = 0

            for video_path, label_tensor, meta in zip(video_paths_batch, labels_batch, metas_batch):
                rgb_seq, of_seq = decode_and_prepare_video(
                    video_path=video_path,
                    model_flow=model_flow,
                    aug_norm=args.aug_norm,
                    center_crop_size=(448, 448),
                    device=DEVICE,
                )
                if (rgb_seq is None) or (of_seq is None):
                    # corrupt / empty video
                    continue

                T_i = rgb_seq.shape[0]
                all_rgb.append(rgb_seq)
                all_of.append(of_seq)

                spans.append((cursor, cursor + T_i, int(label_tensor.item()), meta))
                cursor += T_i

            if len(all_rgb) == 0:
                continue  # nothing valid in this small batch

            # cat into big frame-batch => Option C actual batching
            big_rgb = torch.cat(all_rgb, dim=0)  # [sum_T,3,H,W] on DEVICE
            big_of = torch.cat(all_of, dim=0)  # [sum_T,3,H,W] on DEVICE

            # forward once per modality
            rgb_logits = model_rgb(big_rgb)
            if isinstance(rgb_logits, (list, tuple)):
                rgb_logits = rgb_logits[0]
            rgb_scores_all = torch.sigmoid(rgb_logits).squeeze(dim=-1)  # [sum_T]

            of_logits = model_of(big_of)
            if isinstance(of_logits, (list, tuple)):
                of_logits = of_logits[0]
            of_scores_all = torch.sigmoid(of_logits).squeeze(dim=-1)  # [sum_T]

            # ---- split back per video, aggregate, save rows ----
            for (start_idx, end_idx, gt_label, meta) in spans:
                rgb_scores = rgb_scores_all[start_idx:end_idx]
                of_scores = of_scores_all[start_idx:end_idx]

                combined_scores = (rgb_scores + of_scores) / 2.0  # [T_i]

                rgb_scores_list = rgb_scores.detach().cpu().numpy().tolist()
                of_scores_list = of_scores.detach().cpu().numpy().tolist()
                combined_scores_list = combined_scores.detach().cpu().numpy().tolist()

                video_mean_score = float(np.mean(combined_scores_list))
                pred_label = 1 if video_mean_score >= args.threshold else 0

                row = {
                    "sample_id": meta["sample_id"],
                    "task": meta["task"],
                    "method": meta["method"],
                    "subset": meta["subset"],
                    "label": int(meta["label"]),
                    "model": meta["model"],
                    "mode": meta["mode"],
                    "score": {
                        "rgb_scores": rgb_scores_list,
                        "opt_scores": of_scores_list,
                        "combined_scores": combined_scores_list,
                        "mean_combined": video_mean_score,
                    },
                    "pred": int(pred_label),
                }

                df_row = pd.DataFrame([row])
                if sub_csv_path is not None:
                    df_row.to_csv(
                        sub_csv_path,
                        mode='a',
                        header=not os.path.exists(sub_csv_path),
                        index=False
                    )
                else:
                    df_row.to_csv(
                        args.pred_csv,
                        mode='a',
                        header=not os.path.exists(args.pred_csv),
                        index=False
                    )
    # with torch.no_grad():
    #     for batch in tqdm(loader, dynamic_ncols=True, desc="Evaluating videos"):
    #         if batch is None:
    #             continue  # all vids in this loader batch were corrupt
    #         if fail_count > fail_threshold:
    #             raise RuntimeError(f"Too many failures (> {fail_threshold}), aborting.")
    #
    #         vids_batch, labels_batch, metas_batch = batch
    #         # vids_batch: list[Tensor[T,3,H,W]]
    #         # labels_batch: Tensor[B]
    #         # metas_batch: list[dict]
    #
    #         try:
    #             for vid_frames_cpu, label_tensor, meta in zip(vids_batch, labels_batch, metas_batch):
    #                 # Prepare RGB frames
    #                 # vid_frames_cpu: [T,3,H,W] on CPU, range [0,1]
    #                 if args.aug_norm:
    #                     # ImageNet norm for RGB branch
    #                     vid_frames_norm = TF.normalize(
    #                         vid_frames_cpu,
    #                         mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225],
    #                     )
    #                 else:
    #                     vid_frames_norm = vid_frames_cpu
    #
    #                 vid_frames_norm = vid_frames_norm.to(DEVICE, non_blocking=True)  # [T,3,H,W]
    #
    #                 T, _, H, W = vid_frames_norm.shape
    #
    #                 # Build Optical Flow frames (RAFT → colour wheel → tensor)
    #                 of_tensors: List[torch.Tensor] = []
    #                 for t in range(T):
    #                     if t < T - 1:
    #                         img1 = vid_frames_norm[t:t + 1]  # [1,3,H,W]
    #                         img2 = vid_frames_norm[t + 1:t + 2]
    #                     else:
    #                         # last frame: duplicate prev pair
    #                         img1 = vid_frames_norm[t - 1:t]
    #                         img2 = vid_frames_norm[t:t + 1]
    #
    #                     flow_rgb = batch_optical_flow_gen(model_flow, img1, img2)  # np[H,W,3] uint8
    #                     flow_t = torch.from_numpy(flow_rgb).permute(2, 0, 1).float()  # [3,H,W], 0..255
    #
    #                     # map to [-1,1] like before
    #                     flow_t = (flow_t / 255.0) * 2.0 - 1.0  # [-1,1]
    #                     of_tensors.append(flow_t)
    #
    #                 of_video_cpu = torch.stack(of_tensors, dim=0)  # [T,3,H,W] float32 on CPU
    #
    #                 if args.aug_norm:
    #                     # simple norm for optical flow RGB visualisation
    #                     of_video_norm = TF.normalize(
    #                         of_video_cpu,
    #                         mean=[0.5, 0.5, 0.5],
    #                         std=[0.226, 0.226, 0.226],
    #                     )
    #                 else:
    #                     of_video_norm = of_video_cpu
    #
    #                 of_video_norm = of_video_norm.to(DEVICE, non_blocking=True)  # [T,3,H,W]
    #
    #                 # Forward pass per-frame
    #                 rgb_logits = model_rgb(vid_frames_norm)
    #                 if isinstance(rgb_logits, (list, tuple)):
    #                     rgb_logits = rgb_logits[0]
    #                 rgb_scores = torch.sigmoid(rgb_logits).squeeze(dim=-1)  # [T]
    #
    #                 of_logits = model_of(of_video_norm)
    #                 if isinstance(of_logits, (list, tuple)):
    #                     of_logits = of_logits[0]
    #                 of_scores = torch.sigmoid(of_logits).squeeze(dim=-1)  # [T]
    #
    #                 # combine score per frame
    #                 combined_scores = (rgb_scores + of_scores) / 2.0  # [T]
    #
    #                 # to python lists
    #                 rgb_scores_list = rgb_scores.detach().cpu().numpy().tolist()
    #                 of_scores_list = of_scores.detach().cpu().numpy().tolist()
    #                 combined_scores_list = combined_scores.detach().cpu().numpy().tolist()
    #
    #                 # per-video aggregate for final decision
    #                 video_mean_score = float(np.mean(combined_scores_list))
    #                 pred_label = 1 if video_mean_score >= args.threshold else 0
    #
    #                 # Save row for this video
    #                 # score column stores all 3 arrays as requested
    #                 row = {
    #                     "sample_id": meta["sample_id"],
    #                     "task": meta["task"],
    #                     "method": meta["method"],
    #                     "subset": meta["subset"],
    #                     "label": int(meta["label"]),  # GT
    #                     "model": meta["model"],  # model name/id
    #                     "mode": meta["mode"],  # 'video'
    #                     "score": {
    #                         "rgb_scores": rgb_scores_list,
    #                         "opt_scores": of_scores_list,
    #                         "combined_scores": combined_scores_list,
    #                         "mean_combined": video_mean_score,
    #                     },
    #                     "pred": int(pred_label),
    #                 }
    #                 df_row = pd.DataFrame([row])
    #                 if sub_csv_path is not None:
    #                     # save to per-rank csv
    #                     df_row.to_csv(sub_csv_path, mode='a', header=not os.path.exists(sub_csv_path), index=False)
    #                 else:
    #                     # save to final csv
    #                     df_row.to_csv(args.pred_csv, mode='a', header=not os.path.exists(args.pred_csv), index=False)
    #         except Exception as e:
    #             tqdm.write(str(e), file=sys.stderr)
    #             continue

    if sub_csv_path is not None:
        # combine per-rank csvs
        combined_df = pd.concat([pd.read_csv(f"{args.data_csv[:-4]}_rank{r}.csv") for r in range(world_size)])
        combined_df.to_csv(args.pred_csv, index=False)
        # remove sub csvs
        for r in range(world_size):
            os.remove(f"{args.data_csv[:-4]}_rank{r}.csv")


if __name__ == '__main__':
    main()
