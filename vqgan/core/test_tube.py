# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Tube Logger
----------------
"""
import argparse
import functools
import operator
import numpy as np
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from abc import ABC, abstractmethod
import torch

try:
    from test_tube import Experiment
except ImportError:  # pragma: no-cover
    Experiment = None

from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


def rank_zero_experiment(fn: Callable) -> Callable:
    """ Returns the real experiment on rank 0 and otherwise the DummyExperiment. """
    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyExperiment()
    return experiment


class DummyExperiment(object):
    """ Dummy experiment """
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx):
        # enables self.logger[0].experiment.add_image
        # and self.logger.experiment[0].add_image(...)
        return self


class LightningLoggerBase(ABC):
    """
    Base class for experiment loggers.

    Args:
        agg_key_funcs:
            Dictionary which maps a metric name to a function, which will
            aggregate the metric values for the same steps.
        agg_default_func:
            Default function to aggregate metric values. If some metric name
            is not presented in the `agg_key_funcs` dictionary, then the
            `agg_default_func` will be used for aggregation.

    Note:
        The `agg_key_funcs` and `agg_default_func` arguments are used only when
        one logs metrics with the :meth:`~LightningLoggerBase.agg_and_log_metrics` method.
    """

    def __init__(
            self,
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Callable[[Sequence[float]], float] = np.mean
    ):
        self._prev_step: int = -1
        self._metrics_to_agg: List[Dict[str, float]] = []
        self._agg_key_funcs = agg_key_funcs if agg_key_funcs else {}
        self._agg_default_func = agg_default_func

    def update_agg_funcs(
            self,
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Callable[[Sequence[float]], float] = np.mean
    ):
        """
        Update aggregation methods.

        Args:
            agg_key_funcs:
                Dictionary which maps a metric name to a function, which will
                aggregate the metric values for the same steps.
            agg_default_func:
                Default function to aggregate metric values. If some metric name
                is not presented in the `agg_key_funcs` dictionary, then the
                `agg_default_func` will be used for aggregation.
        """
        if agg_key_funcs:
            self._agg_key_funcs.update(agg_key_funcs)
        if agg_default_func:
            self._agg_default_func = agg_default_func

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger."""

    def _aggregate_metrics(
            self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> Tuple[int, Optional[Dict[str, float]]]:
        """
        Aggregates metrics.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded

        Returns:
            Step and aggregated metrics. The return value could be ``None``. In such case, metrics
            are added to the aggregation list, but not aggregated yet.
        """
        # if you still receiving metric from the same step, just accumulate it
        if step == self._prev_step:
            self._metrics_to_agg.append(metrics)
            return step, None

        # compute the metrics
        agg_step, agg_mets = self._reduce_agg_metrics()

        # as new step received reset accumulator
        self._metrics_to_agg = [metrics]
        self._prev_step = step
        return agg_step, agg_mets

    def _reduce_agg_metrics(self):
        """Aggregate accumulated metrics."""
        # compute the metrics
        if not self._metrics_to_agg:
            agg_mets = None
        elif len(self._metrics_to_agg) == 1:
            agg_mets = self._metrics_to_agg[0]
        else:
            agg_mets = merge_dicts(self._metrics_to_agg, self._agg_key_funcs, self._agg_default_func)
        return self._prev_step, agg_mets

    def _finalize_agg_metrics(self):
        """This shall be called before save/close."""
        agg_step, metrics_to_log = self._reduce_agg_metrics()
        self._metrics_to_agg = []

        if metrics_to_log is not None:
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Aggregates and records metrics.
        This method doesn't log the passed metrics instantaneously, but instead
        it aggregates them and logs only if metrics are ready to be logged.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        agg_step, metrics_to_log = self._aggregate_metrics(metrics=metrics, step=step)

        if metrics_to_log:
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Records metrics.
        This method logs metrics as as soon as it received them. If you want to aggregate
        metrics for one specific `step`, use the
        :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @staticmethod
    def _sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

        Args:
            params: Dictionary containing the hyperparameters

        Returns:
            dictionary with all callables sanitized
        """
        def _sanitize_callable(val):
            # Give them one chance to return a value. Don't go rabbit hole of recursive call
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            return val

        return {key: _sanitize_callable(val) for key, val in params.items()}

    @staticmethod
    def _flatten_dict(params: Dict[str, Any], delimiter: str = '/') -> Dict[str, Any]:
        """
        Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

        Args:
            params: Dictionary containing the hyperparameters
            delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

        Returns:
            Flattened dict.

        Examples:
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, MutableMapping):
                for key, value in input_dict.items():
                    if isinstance(value, (MutableMapping, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns params with non-primitvies converted to strings for logging.

        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(LightningLoggerBase._sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
         'float': 0.3,
         'int': 1,
         'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
         'list': '[1, 2, 3]',
         'namespace': 'Namespace(foo=3)',
         'string': 'abc'}
        """
        for k in params.keys():
            # convert relevant np scalars to python types first (instead of str)
            if isinstance(params[k], (np.bool_, np.integer, np.floating)):
                params[k] = params[k].item()
            elif type(params[k]) not in [bool, int, float, str, torch.Tensor]:
                params[k] = str(params[k])
        return params

    @abstractmethod
    def log_hyperparams(self, params: argparse.Namespace):
        """
        Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
        """

    def log_graph(self, model: LightningModule, input_array=None) -> None:
        """
        Record model graph

        Args:
            model: lightning model
            input_array: input passes to `model.forward`
        """
        pass

    def save(self) -> None:
        """Save log data."""
        self._finalize_agg_metrics()

    def finalize(self, status: str) -> None:
        """
        Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()

    def close(self) -> None:
        """Do any cleanup that is necessary to close an experiment."""
        self.save()

    @property
    def save_dir(self) -> Optional[str]:
        """
        Return the root directory where experiment logs get saved, or `None` if the logger does not
        save data locally.
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Union[int, str]:
        """Return the experiment version."""

    def _add_prefix(self, metrics: Dict[str, float]):
        if self._prefix:
            metrics = {f'{self._prefix}{self.LOGGER_JOIN_CHAR}{k}': v for k, v in metrics.items()}

        return metrics


def merge_dicts(
        dicts: Sequence[Mapping],
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        default_func: Callable[[Sequence[float]], float] = np.mean
) -> Dict:
    """
    Merge a sequence with dictionaries into one dictionary by aggregating the
    same keys with some given function.

    Args:
        dicts:
            Sequence of dictionaries to be merged.
        agg_key_funcs:
            Mapping from key name to function. This function will aggregate a
            list of values, obtained from the same key of all dictionaries.
            If some key has no specified aggregation function, the default one
            will be used. Default is: ``None`` (all keys will be aggregated by the
            default function).
        default_func:
            Default function to aggregate keys, which are not presented in the
            `agg_key_funcs` map.

    Returns:
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1, 'd': {'d1': 1, 'd3': 3}}
        >>> d2 = {'a': 1.1, 'b': 2.2, 'v': 1, 'd': {'d1': 2, 'd2': 3}}
        >>> d3 = {'a': 1.1, 'v': 2.3, 'd': {'d3': 3, 'd4': {'d5': 1}}}
        >>> dflt_func = min
        >>> agg_funcs = {'a': np.mean, 'v': max, 'd': {'d1': sum}}
        >>> pprint.pprint(merge_dicts([d1, d2, d3], agg_funcs, dflt_func))
        {'a': 1.3,
         'b': 2.0,
         'c': 1,
         'd': {'d1': 3, 'd2': 3, 'd3': 3, 'd4': {'d5': 1}},
         'v': 2.3}
    """
    agg_key_funcs = agg_key_funcs or dict()
    keys = list(functools.reduce(operator.or_, [set(d.keys()) for d in dicts]))
    d_out = {}
    for k in keys:
        fn = agg_key_funcs.get(k)
        values_to_agg = [v for v in [d_in.get(k) for d_in in dicts] if v is not None]

        if isinstance(values_to_agg[0], dict):
            d_out[k] = merge_dicts(values_to_agg, fn, default_func)
        else:
            d_out[k] = (fn or default_func)(values_to_agg)

    return d_out


class TestTubeLogger(LightningLoggerBase):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format
    but using a nicer folder structure (see `full docs <https://williamfalcon.github.io/test-tube>`_).

    Install it with pip:

    .. code-block:: bash

        pip install test_tube

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import TestTubeLogger
        logger = TestTubeLogger("tt_logs", name="my_exp_name")
        trainer = Trainer(logger=logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from pytorch_lightning import LightningModule
        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_method_summary_writer_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.add_histogram(...)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        description: A short snippet about this experiment
        debug: If ``True``, it doesn't log anything.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        create_git_tag: If ``True`` creates a git tag to save the code used in this experiment.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        prefix: A string to put at the beginning of metric keys.
    """

    __test__ = False
    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self,
        save_dir: str,
        name: str = "default",
        description: Optional[str] = None,
        debug: bool = False,
        version: Optional[int] = None,
        create_git_tag: bool = False,
        log_graph: bool = False,
        prefix: str = '',
    ):
        if Experiment is None:
            raise ImportError('You want to use `test_tube` logger which is not installed yet,'
                              ' install it with `pip install test-tube`.')
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self.description = description
        self.debug = debug
        self._version = version
        self.create_git_tag = create_git_tag
        self._log_graph = log_graph
        self._prefix = prefix
        self._experiment = None

    @property
    @rank_zero_only
    def experiment(self) -> Experiment:
        r"""

        Actual TestTube object. To use TestTube features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_test_tube_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=rank_zero_only.rank,
        )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self.experiment.argparse(Namespace(**params))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # TODO: HACK figure out where this is being set to true
        metrics = self._add_prefix(metrics)
        self.experiment.debug = self.debug
        self.experiment.log(metrics, global_step=step)

    @rank_zero_only
    def log_graph(self, model: LightningModule, input_array=None):
        if self._log_graph:
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                self.experiment.add_graph(
                    model,
                    model.transfer_batch_to_device(
                        model.example_input_array, model.device)
                )
            else:
                rank_zero_warn('Could not log computational graph since the'
                               ' `model.example_input_array` attribute is not set'
                               ' or `input_array` was not given',
                               UserWarning)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.save()
        self.close()

    @rank_zero_only
    def close(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        if not self.debug:
            exp = self.experiment
            exp.close()

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> str:
        if self._experiment is None:
            return self._name
        else:
            return self.experiment.name

    @property
    def version(self) -> int:
        if self._experiment is None:
            return self._version
        else:
            return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state: Dict[Any, Any]):
        self._experiment = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)
