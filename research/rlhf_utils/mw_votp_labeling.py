import os
import copy
import argparse
import numpy as np
import ot
from tqdm import tqdm
from termcolor import cprint
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Define constants
# Assume the order of pair (seg_1, seg_2) ~ (left segment, right segment)
PREF_LEFT = -1.0  # seg_1 > seg_2
PREF_RIGHT = 1.0  # seg_1 < seg_2
PREF_EQUAL = 0.0  # seg_1 = seg_2


def check_relation_matrix(dense_labels):
    for i in range(dense_labels.shape[0]):
        for j in range(dense_labels.shape[1]):
            if i == j:
                assert dense_labels[i, j] == PREF_EQUAL
            if dense_labels[i, j] == PREF_RIGHT:
                assert dense_labels[j, i] == PREF_LEFT
            elif dense_labels[i, j] == PREF_LEFT:
                assert dense_labels[j, i] == PREF_RIGHT
            elif dense_labels[i, j] == PREF_EQUAL:
                assert dense_labels[j, i] == PREF_EQUAL


class VOTP_PseudoLabeler():
    def __init__(self, n_src_pairs, src_1_feats, src_2_feats, src_pref_labels=None, entropy_reg=0.01):
        self.n_src_pairs = n_src_pairs
        self.n_src_samples = n_src_pairs * 2
        self.n_tgt_samples = 2
        self.entropy_reg = entropy_reg

        assert src_1_feats.shape[0] == n_src_pairs and src_2_feats.shape[0] == n_src_pairs

        self.P = ot.unif(self.n_src_samples)   # distribution over source samples
        self.Q = ot.unif(self.n_tgt_samples)   # distribution over target samples

        self.src_feats = np.concatenate([src_1_feats, src_2_feats], axis=0)
        relation_matrix = np.zeros((self.n_src_samples, self.n_src_samples), dtype=np.float32)

        if src_pref_labels is not None:
            assert src_pref_labels.shape[0] == n_src_pairs
            for left_idx, right_idx in zip(range(0, self.n_src_pairs), range(self.n_src_pairs, self.n_src_pairs * 2)):
                # Note that the label here is in [0, 1, 0.5]
                if src_pref_labels[left_idx] == 0.0:
                    relation_matrix[left_idx, right_idx] = PREF_LEFT
                    relation_matrix[right_idx, left_idx] = PREF_RIGHT

                elif src_pref_labels[left_idx] == 1.0:
                    relation_matrix[left_idx, right_idx] = PREF_RIGHT
                    relation_matrix[right_idx, left_idx] = PREF_LEFT

                elif src_pref_labels[left_idx] == 0.5:
                    relation_matrix[left_idx, right_idx] = PREF_EQUAL
                    relation_matrix[right_idx, left_idx] = PREF_EQUAL

                else:
                    raise ValueError(f"Unknown preference label: {src_pref_labels[left_idx]}")

        else:
            relation_matrix[:n_src_pairs, n_src_pairs:] = PREF_LEFT
            relation_matrix[n_src_pairs:, :n_src_pairs] = PREF_RIGHT

        check_relation_matrix(relation_matrix)

        self.relation_matrix = relation_matrix
        self.max_pref_score = self.compute_norm_preference_score(self.relation_matrix)

    def compute_norm_preference_score(self, relation_matrix):
        # Based on mass of source samples
        S_max = 0.0
        for i in range(self.n_src_pairs * 2):
            for j in range(self.n_src_pairs * 2):
                if relation_matrix[i, j] != 0:
                    S_max += (1 / self.n_src_samples) * (1 / self.n_src_samples)
        return S_max

    def compute_preference_score(self, tgt_1_feats, tgt_2_feats):
        assert tgt_1_feats.shape[1] == self.src_feats.shape[1] and tgt_2_feats.shape[1] == self.src_feats.shape[1]
        tgt_feats = np.concatenate([tgt_1_feats, tgt_2_feats], axis=0)

        # Step 1: Compute cost matrix
        cost_matrix = cdist(self.src_feats, tgt_feats, metric='euclidean')

        # Step 2: Compute OT plan
        if self.entropy_reg == 0:
            coupling_matrix = ot.emd(self.P, self.Q, cost_matrix)
        else:
            coupling_matrix = ot.sinkhorn(self.P, self.Q, cost_matrix, reg=self.entropy_reg, method="sinkhorn", numItermax=2000)

        # Step 3: Compute the preference score by computing direction of mass relative to source pairs
        coupling_matrix[coupling_matrix < 1e-6] = 0  # post-process to avoid numerical error
        coupling_matrix = coupling_matrix.astype(np.float32)

        # pref = Σ_ij R[i,j] (C[i,si]C[j,sj] − C[i,sj]C[j,si]) = a·R·b − b·R·a
        a = coupling_matrix[:, 0]
        b = coupling_matrix[:, 1]
        pref_score = float(a @ self.relation_matrix @ b - b @ self.relation_matrix @ a)

        pref_score = pref_score / self.max_pref_score
        return pref_score


def mw_votp_labeling(
        env_name,
        entropic_reg,
        preference_thresh=0.0,
        equal_pref_threshold_teacher=0.0,
        n_labels=10,
        max_pairs=50000,
        return_training_data=False,
        verbose=True,
        save_pseudo_preference=False,
        vifm="s3d",
        use_gt_only=False
):
    # Define relative path
    pseudo_preference_path = "datasets/metaworld/pseudo_preferences"
    pair_indices_path = f"datasets/metaworld/pseudo_preferences/pair_indices/{env_name}_pair_indices_{max_pairs}.npz"
    segment_data_file = f"datasets/metaworld/segments/data/{env_name}_nseg20000_size64.npz"
    segment_feat_file = f"datasets/metaworld/segments/features/{env_name}_corner2_v2_feat_{vifm}.npz"
    assert os.path.exists(segment_data_file) and os.path.exists(segment_feat_file), f"Missing files: {segment_data_file}, {segment_feat_file}"

    # Segments [0, N_SEG_PER_SIDE) are left candidates; [N_SEG_PER_SIDE, 2*N_SEG_PER_SIDE) are right (in segment_feat)
    N_SEG_PER_SIDE = 10000  # Do not change

    # Prepare features and extra information for data pair
    segment_data = np.load(segment_data_file, allow_pickle=True)
    segment_feat = np.load(segment_feat_file, allow_pickle=True)["feature"]

    # Load pair indices
    assert os.path.exists(pair_indices_path), f"File is not exist: {pair_indices_path}"
    _tmp = np.load(pair_indices_path, allow_pickle=True)
    pool_seg_1_idxes, pool_seg_2_idxes = _tmp["pool_seg_1_idxes"], _tmp["pool_seg_2_idxes"]

    # Prepare ground-truth preferences: reward is already normalized to [0, 1] per step
    preference_teacher = segment_data["reward"].sum(axis=1)  # Sum along the segment

    # Equal-preference margin: a pair is "equal" if |R1 - R2| <= segment_size * threshold (following LiRE, BPref)
    equal_margin = segment_data["reward"].shape[1] * equal_pref_threshold_teacher

    # Ground-truth preference for ALL pairs, in {-1, 0, 1} space
    return_diff = preference_teacher[pool_seg_1_idxes] - preference_teacher[pool_seg_2_idxes + N_SEG_PER_SIDE]
    is_equal = np.abs(return_diff) <= equal_margin
    gt_preferences = np.where(is_equal, PREF_EQUAL, np.where(return_diff > 0, PREF_LEFT, PREF_RIGHT))

    # Construct indices for labeled and unlabeled pairs
    labeled_idxes = np.arange(0, n_labels)
    unlabeled_idxes = np.setdiff1d(np.arange(max_pairs), labeled_idxes)

    # Features and preference labels for labeled pairs
    # Map GT {-1, 0, 1} -> labeler convention {0, 1, 0.5}: PREF_LEFT->0, PREF_RIGHT->1, PREF_EQUAL->0.5
    gt_labeled = gt_preferences[labeled_idxes]
    src_pref_labels = np.where(gt_labeled == PREF_LEFT, 0.0, np.where(gt_labeled == PREF_RIGHT, 1.0, 0.5))
    src_1_feats = segment_feat[pool_seg_1_idxes[labeled_idxes]]
    src_2_feats = segment_feat[pool_seg_2_idxes[labeled_idxes] + N_SEG_PER_SIDE]
    labeled_pos = {int(p): k for k, p in enumerate(labeled_idxes)}  # global pair idx -> position in src labels

    if use_gt_only:
        pair_indices = []
        for idx in range(max_pairs):
            seg_1_idx, seg_2_idx = pool_seg_1_idxes[idx], pool_seg_2_idxes[idx] + N_SEG_PER_SIDE
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
        cprint(f"{env_name}: #labels={n_labels}, #pairs={max_pairs}, ent_reg={entropic_reg}, "
               f"p_thr={preference_thresh}, eq_thr={equal_pref_threshold_teacher}, "
               f"#strict={int((~is_equal).sum())}/{max_pairs}", "green")

    # Generate pseudo preference
    norm_preference_scores, pair_indices = [], []
    for idx in tqdm(range(max_pairs), leave=False, desc="VOTP labeling"):
        seg_1_idx, seg_2_idx = pool_seg_1_idxes[idx], pool_seg_2_idxes[idx] + N_SEG_PER_SIDE

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

    if save_pseudo_preference:
        filename = os.path.join(
            pseudo_preference_path,
            f"{env_name}_N{n_labels}_M{max_pairs}_reg{entropic_reg}_eq{equal_pref_threshold_teacher}_script_{vifm}.npz"
        )

        np.savez(
            filename,
            pair_indices=pair_indices,
            norm_preference_scores=norm_preference_scores,
            gt_preferences=gt_preferences,
            meta_data={
                "n_labels": n_labels,
                "max_pairs": max_pairs,
                "entropic_reg": entropic_reg,
                "equal_pref_threshold_teacher": equal_pref_threshold_teacher,
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
        # Evaluate ONLY where GT is also strict and exclude labeled source pairs
        labeled_mask = np.zeros(max_pairs, dtype=bool)
        labeled_mask[labeled_idxes] = True
        eval_mask = retained_mask & (gt_preferences != PREF_EQUAL) & ~labeled_mask

        # Map the evaluated subset from {-1, 1} -> {0, 1} so sklearn binary metrics are always valid
        y_pred = ((pseudo_pref[eval_mask] + 1) / 2).astype(int)
        y_true = ((gt_preferences[eval_mask] + 1) / 2).astype(int)
        retained_pair_indices = pair_indices[retained_mask]

        min_p_score = norm_preference_scores[unlabeled_idxes].min()
        max_p_score = norm_preference_scores[unlabeled_idxes].max()

        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100

        cprint(
            f"[{env_name}] VOTP Labeling:\n"
            f"Preference score range: ({min_p_score:.2f}, {max_p_score:.2f})\n"
            f"Preference threshold: {preference_thresh}, retained pairs: {len(retained_pair_indices)} "
            f"(evaluated on {len(y_true)} strict-GT pairs)\n"
            f"Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}\n",
            'yellow', attrs=['bold']
        )

        return {
            "pseudo_labels": ((pseudo_pref[retained_mask] + 1) / 2),    # in {0, 1} space
            "retained_pair_indices": pair_indices[retained_mask],
            "gt_preferences": (gt_preferences + 1) / 2,        # GT for all pairs, in {0, 1, 0.5} space
            "pair_indices": pair_indices,  # all sampled pairs
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
    args = parser.parse_args()

    mw_votp_labeling(
        env_name=args.env_name,
        entropic_reg=args.entropic_reg,
        preference_thresh=args.preference_thresh,
        equal_pref_threshold_teacher=args.equal_pref_threshold_teacher,
        return_training_data=True,
        save_pseudo_preference=args.save_pseudo_preference,
    )
