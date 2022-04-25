import torch


COLLECT_STATS = False

def reset_stats(input_obj):
    for attr_list in ["_record_these_stats"]:
        for r in getattr(input_obj, attr_list, []):
            setattr(input_obj, r, 0)

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def add_to_recordable_attributes(
    input_obj, name=None, list_of_names=None, is_stat=False
):
    if is_stat:
        attr_name_list_name = "_record_these_stats"
    else:
        attr_name_list_name = "_record_these"
    if not hasattr(input_obj, attr_name_list_name):
        setattr(input_obj, attr_name_list_name, [])
    attr_name_list = getattr(input_obj, attr_name_list_name)
    if name is not None:
        if name not in attr_name_list:
            attr_name_list.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)
    if list_of_names is not None and is_list_or_tuple(list_of_names):
        for n in list_of_names:
            add_to_recordable_attributes(input_obj, name=n, is_stat=is_stat)

class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = (
            COLLECT_STATS if collect_stats is None else collect_stats
        )

    def add_to_recordable_attributes(
        self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )

    def reset_stats(self):
        reset_stats(self)

class BaseDistance(ModuleWithRecords):
    def __init__(
        self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(
        self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            with torch.no_grad():
                stats_dict = {
                    "initial_avg_query_norm": torch.mean(
                        self.get_norm(query_emb)
                    ).item(),
                    "initial_avg_ref_norm": torch.mean(self.get_norm(ref_emb)).item(),
                    "final_avg_query_norm": torch.mean(
                        self.get_norm(query_emb_normalized)
                    ).item(),
                    "final_avg_ref_norm": torch.mean(
                        self.get_norm(ref_emb_normalized)
                    ).item(),
                }
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)

def meshgrid_from_sizes(x, y, dim=0):
    a = torch.arange(x.size(dim), device=x.device)
    b = torch.arange(y.size(dim), device=y.device)
    return torch.meshgrid(a, b, indexing="ij")

class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)

class ModuleWithRecordsAndDistance(ModuleWithRecords):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2)

###############

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if not torch.is_tensor(ref_labels):
            TypeError("if ref_emb is given, then ref_labels must also be given")
        ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, ref_labels

def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x

def check_shapes(embeddings, labels):
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
        )
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")

#################

class BaseMiner(ModuleWithRecordsAndDistance):
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        raise NotImplementedError

    def output_assertion(self, output):
        raise NotImplementedError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            check_shapes(embeddings, labels)
            labels = to_device(labels, embeddings)
            ref_emb, ref_labels = set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output


class BaseTupleMiner(BaseMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(
            list_of_names=["num_pos_pairs", "num_neg_pairs", "num_triplets"],
            is_stat=True,
        )

    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices or a1,p,a2,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        elif len(output) == 4:
            self.num_pos_pairs = len(output[0])
            self.num_neg_pairs = len(output[2])
            assert self.num_pos_pairs == len(output[1])
            assert self.num_neg_pairs == len(output[3])
        else:
            raise TypeError

#################

def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs

def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)

#################

class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()