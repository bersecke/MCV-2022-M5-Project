import torch 
import inspect
import copy

COLLECT_STATS = False

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

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if not torch.is_tensor(ref_labels):
            TypeError("if ref_emb is given, then ref_labels must also be given")
        ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, 

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

class EmbeddingRegularizerMixin:
    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        self.embedding_regularizer = (
            embedding_regularizer is not None
        )  # hack needed to know whether reg will be in sub-loss names
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight
        if self.embedding_regularizer is not None:
            self.add_to_recordable_attributes(
                list_of_names=["embedding_reg_weight"], is_stat=False
            )

    def embedding_regularization_loss(self, embeddings):
        if self.embedding_regularizer is None:
            loss = 0
        else:
            loss = self.embedding_regularizer(embeddings) * self.embedding_reg_weight
        return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        if self.embedding_regularizer is not None:
            loss_dict["embedding_reg_loss"] = self.embedding_regularization_loss(
                embeddings
            )

    def regularization_loss_names(self):
        return ["embedding_reg_loss"]

class BaseReducer(ModuleWithRecords):
    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        self.add_to_recordable_attributes(name=loss_name, is_stat=True)
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        setattr(self, loss_name, loss_val.item())
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            self.add_to_recordable_attributes(name="losses_size", is_stat=True)
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)

class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(
            len(loss_dict), dtype=embeddings.dtype, device=embeddings.device
        )
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)

class DoNothingReducer(BaseReducer):
    def forward(self, loss_dict, embeddings, labels):
        return loss_dict

class ModuleWithRecordsAndReducer(ModuleWithRecords):
    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_reducer(reducer)

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        if isinstance(reducer, (MultipleReducers, DoNothingReducer)):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = (
                self.get_default_reducer()
                if reducer is None
                else copy.deepcopy(reducer)
            )
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = (
                    self.get_default_reducer()
                    if reducer is None
                    else copy.deepcopy(reducer)
                )
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ["loss"]

def reset_stats(input_obj):
    for attr_list in ["_record_these_stats"]:
        for r in getattr(input_obj, attr_list, []):
            setattr(input_obj, r, 0)

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

class ModuleWithRecordsReducerAndDistance(
    ModuleWithRecordsAndReducer, ModuleWithRecordsAndDistance
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BaseMetricLossFunction(
    EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance
):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(
        self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        check_shapes(embeddings, labels)
        labels = to_device(labels, embeddings)
        ref_emb, ref_labels = set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = "RegularizerMixin"
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, "").lower()
                if getattr(self, "{}_regularizer".format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names

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



class BaseReducer(ModuleWithRecords):
    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        self.add_to_recordable_attributes(name=loss_name, is_stat=True)
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        setattr(self, loss_name, loss_val.item())
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            self.add_to_recordable_attributes(name="losses_size", is_stat=True)
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)

class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (
            high is not None
        ), "At least one of low or high must be specified"
        self.low = low
        self.high = high
        if self.low is not None:
            self.add_to_recordable_attributes(list_of_names=["low"], is_stat=False)
        if self.high is not None:
            self.add_to_recordable_attributes(list_of_names=["high"], is_stat=False)

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "elements")

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "pos_pairs")

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "neg_pairs")

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "triplets")

    def element_reduction_helper(self, losses, embeddings, attr_name):
        low_condition = losses > self.low if self.low is not None else True
        high_condition = losses < self.high if self.high is not None else True
        threshold_condition = low_condition & high_condition
        num_past_filter = torch.sum(threshold_condition)
        if num_past_filter >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter, attr_name)
        return loss

    def set_stats(self, low_condition, high_condition, num_past_filter, attr_name):
        if self.collect_stats:
            curr_attr_name = "{}_past_filter".format(attr_name)
            self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
            setattr(self, curr_attr_name, num_past_filter.item())
            with torch.no_grad():
                if self.low is not None:
                    curr_attr_name = "{}_above_low".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(low_condition).item())
                if self.high is not None:
                    curr_attr_name = "{}_below_high".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(high_condition).item())

class AvgNonZeroReducer(ThresholdReducer):
    def __init__(self, **kwargs):
        super().__init__(low=0, **kwargs)

def get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = torch.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
        n_inds = torch.where(ref_labels != label)[0]
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = to_device(torch.cat(a_idx), device=labels_device, dtype=torch.long)
        p_idx = to_device(torch.cat(p_idx), device=labels_device, dtype=torch.long)
        n_idx = to_device(torch.cat(n_idx), device=labels_device, dtype=torch.long)
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.tensor([], device=labels_device, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone()

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

def convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return get_random_triplet_indices(
                labels, ref_labels, t_per_anchor=t_per_anchor
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]

class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        indices_tuple = convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)
        violation = current_margins + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()