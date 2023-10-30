import itertools
import more_itertools

from collections import defaultdict
from typing import Any, Dict, Iterable, Union, List, Mapping

from allennlp.data.instance import Instance
from allennlp.data.data_loaders.multitask_scheduler import (
    MultiTaskScheduler,
    _chunked_iterator,
)


@MultiTaskScheduler.register("unbalanced_homogeneous_roundrobin")
class HomogeneousRoundRobinScheduler(MultiTaskScheduler):
    """
    Orders instances in a round-robin fashion, but grouped into batches composed entirely of
    instances from one dataset.  We'll return one batch from one dataset, then another batch from a
    different dataset, etc.  This is currently necessary in AllenNLP if your instances have
    different fields for different datasets, as we can't currently combine instances with different
    fields.
    When one dataset runs out, we continue iterating round-robin through the rest.
    If you want more fine-grained control over which datasets can be combined, it should be
    relatively straightforward to write your own scheduler, following this logic, which allows some
    datasets to be combined and others not.
    Registered as a `MultiTaskScheduler` with name "homogeneous_roundrobin".
    # Parameters
    batch_size: `Union[int, Dict[str, int]]`
        Determines how many instances to group together in each dataset.  If this is an `int`, the
        same value is used for all datasets; otherwise, the keys must correspond to the dataset
        names used elsewhere in the multi-task code.
    """

    def __init__(
        self,
        batch_size: Union[int, Dict[str, int]],
        drop_last: bool = False,
        dataset_sizes: Dict[str, int] = None,
    ):
        self.batch_size: Mapping[str, int]
        if isinstance(batch_size, int):
            self.batch_size = defaultdict(lambda: batch_size)  # type: ignore
        else:
            self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_sizes = dataset_sizes
        self.max_dataset_size = (
            max(dataset_sizes.values()) if dataset_sizes else -1
        )

    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        chunked_iterators = [
            _chunked_iterator(
                iterator, self.batch_size[dataset], self.drop_last
            )
            if self.dataset_sizes[dataset] >= self.max_dataset_size
            else itertools.cycle(
                _chunked_iterator(
                    iterator, self.batch_size[dataset], self.drop_last
                )
            )
            for dataset, iterator in epoch_instances.items()
        ]
        return more_itertools.interleave(*chunked_iterators)

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        # result = 0
        # for dataset, count in dataset_counts.items():
        #     batch_size = self.batch_size[dataset]
        #     result += count // batch_size
        #     if not self.drop_last and count % batch_size != 0:
        #         result += 1
        # return result
        max_base_batches = max(
            count // self.batch_size[dataset]
            + int(not self.drop_last and count % self.batch_size[dataset] != 0)
            for dataset, count in dataset_counts.items()
        )
        return max_base_batches * len(dataset_counts)
