import os
import copy
import argparse
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from research.rlhf_utils.mw_votp_labeling import (
    VOTP_PseudoLabeler, PREF_LEFT, PREF_RIGHT, PREF_EQUAL
)


def loco_votp_labeling(
        env_name,
        entropic_reg,
        preference_thresh=0.0,
        equal_pref_threshold_teacher=0.0,
        n_labels=10,
        max_pairs=10000,
        teacher="script",
        return_training_data=False,
        verbose=True,
        save_pseudo_preference=False,
        vifm="s3d",
        use_gt_only=False
):
    assert teacher in ("human", "script"), f"Unknown teacher: {teacher}"
    assert max_pairs <= 10000, f"Please keep max_pairs <= 10000 for D4RL."

    # Define relative paths
    pseudo_preference_path = "datasets/locomotion/pseudo_preferences"
    os.makedirs(pseudo_preference_path, exist_ok=True)
    segment_data_file = f"datasets/locomotion/segments/data/{env_name}/num10000_{teacher}.npz"
    segment_feat_file = f"datasets/locomotion/segments/features/{env_name}_feat_{vifm}.npz"
    assert os.path.exists(segment_data_file) and os.path.exists(segment_feat_file), f"Missing files: {segment_data_file}, {segment_feat_file}"

    # Segments [0, N_SEG_PER_SIDE) are left candidates; [N_SEG_PER_SIDE, 2*N_SEG_PER_SIDE) are right (in segment_feat)
    N_SEG_PER_SIDE = 10000  # Do not change

    # Prepare features and extra information for data pair
    segment_data = np.load(segment_data_file, allow_pickle=True)
    segment_feat = np.load(segment_feat_file, allow_pickle=True)["feature"]

    # Generate indices for segment pairs
    pool_seg_1_idxes = np.arange(0, max_pairs)
    pool_seg_2_idxes = np.arange(max_pairs, max_pairs * 2)

    # Prepare ground-truth preference
    gt_preferences = np.full(max_pairs, np.nan, dtype=np.float64)
    gt_valid_mask = np.zeros(max_pairs, dtype=bool)     # marks which pairs have GT
    if teacher == "script": # GT from segment reward
        # Equal-preference margin: a pair is "equal" if |R1 - R2| <= segment_size * threshold (following LiRE, BPref)
        equal_margin = segment_data["reward_1"].shape[1] * equal_pref_threshold_teacher

        # Ground-truth preference for ALL pairs, in {-1, 0, 1} space
        return_diff = segment_data["reward_1"][pool_seg_1_idxes].sum(axis=1) - segment_data["reward_2"][pool_seg_2_idxes - N_SEG_PER_SIDE].sum(axis=1)
        is_equal = np.abs(return_diff) <= equal_margin
        gt_preferences = np.where(is_equal, PREF_EQUAL, np.where(return_diff > 0, PREF_LEFT, PREF_RIGHT))

        gt_valid_mask[:] = True
        strict_idxes = np.flatnonzero(~is_equal)
    else:  # human: GT only for the (limited) labeled budget; labels already encode equal (0.5)
        human_labels = segment_data["label"]   # in {0, 0.5, 1} space
        n_human = len(human_labels)
        gt_preferences[:n_human] = np.where(human_labels == 0.5, PREF_EQUAL, np.where(human_labels == 0.0, PREF_LEFT, PREF_RIGHT))

        gt_valid_mask[:n_human] = True
        strict_idxes = np.flatnonzero(gt_valid_mask & (gt_preferences != PREF_EQUAL))

    # Construct indices for labeled and unlabeled pairs
    labeled_idxes = np.arange(0, n_labels)
    unlabeled_idxes = np.setdiff1d(np.arange(max_pairs), labeled_idxes)

    # Features and preference labels for labeled pairs
    # Map GT {-1, 0, 1} -> labeler convention {0, 1, 0.5}: PREF_LEFT->0, PREF_RIGHT->1, PREF_EQUAL->0.5
    gt_labeled = gt_preferences[labeled_idxes]
    src_pref_labels = np.where(gt_labeled == PREF_LEFT, 0.0, np.where(gt_labeled == PREF_RIGHT, 1.0, 0.5))
    src_1_feats = segment_feat[pool_seg_1_idxes[labeled_idxes]]
    src_2_feats = segment_feat[pool_seg_2_idxes[labeled_idxes]]
    labeled_pos = {int(p): k for k, p in enumerate(labeled_idxes)}  # global pair idx -> position in src labels

    if use_gt_only:
        pair_indices = []
        for idx in range(max_pairs):
            seg_1_idx, seg_2_idx = pool_seg_1_idxes[idx], pool_seg_2_idxes[idx]
            pair_indices.append((seg_1_idx, seg_2_idx))
        pair_indices = np.array(pair_indices)
        return {
            "pair_indices": pair_indices,  # all sampled pairs
            "gt_preferences": (gt_preferences + 1) / 2,  # GT for all pairs, in {0, 1, 0.5} space
        }

    # Initialize VOTP Labeler
    preference_labeler = VOTP_PseudoLabeler(
        n_src_pairs=n_labels,
        src_1_feats=src_1_feats, src_2_feats=src_2_feats,
        entropy_reg=entropic_reg,
        src_pref_labels=src_pref_labels
    )

    if verbose:
        cprint(f"{env_name} [{teacher}]: #labels={n_labels}, #pairs={max_pairs}, ent_reg={entropic_reg}, "
               f"p_thr={preference_thresh}, eq_thr={equal_pref_threshold_teacher}, "
               f"#strict={len(strict_idxes)}, #gt={int(gt_valid_mask.sum())}", "green")

    # Generate pseudo preference
    norm_preference_scores, pair_indices = [], []
    for idx in tqdm(range(max_pairs), leave=False, desc="VOTP labeling"):
        seg_1_idx, seg_2_idx = pool_seg_1_idxes[idx], pool_seg_2_idxes[idx]

        # Get preference score from VOTP
        p_score = preference_labeler.compute_preference_score(
            tgt_1_feats=segment_feat[[seg_1_idx]], tgt_2_feats=segment_feat[[seg_2_idx]]
        )

        # Override the p_score if this pair is one of the labeled source pairs
        if idx in labeled_pos:
            lbl = src_pref_labels[labeled_pos[idx]]
            p_score = -1.0 if lbl == 0 else (1.0 if lbl == 1 else 0.0)

        norm_preference_scores.append(p_score)
        pair_indices.append((seg_1_idx, seg_2_idx))

    # Convert all lists to numpy
    pair_indices = np.array(pair_indices)
    norm_preference_scores = np.array(norm_preference_scores)
    # Original segment indices for each pair, might be helful for reward relabeling
    comparison_original_indices = np.stack([segment_data["original_indices_1"][:max_pairs], segment_data["original_indices_2"][:max_pairs]], axis=1)

    if save_pseudo_preference:
        filename = os.path.join(
            pseudo_preference_path,
            f"{env_name}_N{n_labels}_M{max_pairs}_reg{entropic_reg}_eq{equal_pref_threshold_teacher}_{teacher}_{vifm}.npz"
        )

        np.savez(
            filename,
            pair_indices=pair_indices,
            norm_preference_scores=norm_preference_scores,
            gt_preferences=gt_preferences,
            comparison_original_indices=comparison_original_indices,
            meta_data={
                "n_labels": n_labels,
                "max_pairs": max_pairs,
                "entropic_reg": entropic_reg,
                "equal_pref_threshold_teacher": equal_pref_threshold_teacher,
                "teacher": teacher,
                "labeled_pair_idxes": labeled_idxes,
                "encoder_name": vifm,
            },
        )

    if return_training_data:
        # Threshold low-confidence scores to 0 (uncertain -> dropped)
        scores = copy.deepcopy(norm_preference_scores)
        scores[np.abs(scores) < preference_thresh] = 0.0

        pseudo_pref = np.sign(scores)  # in {-1, 0, 1}; 0 = uncertain/equal -> not retained
        retained_mask = pseudo_pref != 0  # confident strict pseudo-labels
        # Evaluate ONLY where GT exists (human budget) and is strict, and exclude the labeled source pairs
        labeled_mask = np.zeros(max_pairs, dtype=bool)
        labeled_mask[labeled_idxes] = True
        eval_mask = retained_mask & gt_valid_mask & (gt_preferences != PREF_EQUAL) & ~labeled_mask

        # Map the evaluated subset from {-1, 1} -> {0, 1} so sklearn binary metrics are always valid
        y_pred = ((pseudo_pref[eval_mask] + 1) / 2).astype(int)
        y_true = ((gt_preferences[eval_mask] + 1) / 2).astype(int)
        retained_pair_indices = pair_indices[retained_mask]

        min_p_score = norm_preference_scores[unlabeled_idxes].min()
        max_p_score = norm_preference_scores[unlabeled_idxes].max()

        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred) * 100
            precision = precision_score(y_true, y_pred, zero_division=0) * 100
            recall = recall_score(y_true, y_pred, zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        else:
            accuracy = precision = recall = f1 = float("nan")

        cprint(
            f"[{env_name}/{teacher}] VOTP Labeling:\n"
            f"Preference score range: ({min_p_score:.2f}, {max_p_score:.2f})\n"
            f"Preference threshold: {preference_thresh}, retained pairs: {len(retained_pair_indices)} "
            f"(evaluated on {len(y_true)} strict-GT pairs)\n"
            f"Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}\n",
            'yellow', attrs=['bold']
        )

        return {
            "pseudo_labels": ((pseudo_pref[retained_mask] + 1) / 2),    # in {0, 1} space
            "retained_pair_indices": retained_pair_indices,
            "gt_preferences": (gt_preferences + 1) / 2,                 # {0, 1, 0.5} (NaN where GT is unavailable)
            "pair_indices": pair_indices,  # all sampled pairs
            "comparison_original_indices": comparison_original_indices,
            "min_p_score": min_p_score, "max_p_score": max_p_score,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        }

    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--entropic_reg", type=float, default=0.01)
    parser.add_argument("--preference_thresh", type=float, default=0.0)
    parser.add_argument("--equal_pref_threshold_teacher", type=float, default=0.0)
    parser.add_argument("--save_pseudo_preference", action="store_true")
    parser.add_argument("--teacher", type=str, default="script", choices=["human", "script"])
    args = parser.parse_args()

    loco_votp_labeling(
        env_name=args.env_name,
        entropic_reg=args.entropic_reg,
        preference_thresh=args.preference_thresh,
        equal_pref_threshold_teacher=args.equal_pref_threshold_teacher,
        teacher=args.teacher,
        return_training_data=True,
        save_pseudo_preference=args.save_pseudo_preference,
    )
