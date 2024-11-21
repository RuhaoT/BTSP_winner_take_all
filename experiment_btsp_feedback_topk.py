"""This experiment replicates and further explores the experiment of Figure 5
in the paper "One-shot learning and robust recall with BTSP, a biological
synaptic plasticity rule" by Yujie Wu, et al. (2024). We intend to explore the
effect of changing the top-k parameter in the BTSP feedback network.
"""

import os
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from experiment_framework_bic.auto_experiment import auto_experiment
from experiment_framework_bic.utils import logging, parameterization

@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment."""

    # experiment parameters
    random_seed: int | list = 0
    precision: str = "half"
    device: str = "cuda"
    experiment_index: int = -1

    # network parameters
    input_size: int | list = 200
    output_dim: int | list = 150
    btsp_fq: float | list = 0.005
    btsp_topk_ratio_nfq: float | list = 0.5
    fw: float | list = 0.6  # connection ratio
    hebbian_topk_ratio_range_mfp: tuple[float, float] | list[tuple[float, float]] = (
        0.5,
        1.5,
    )

    # data parameters
    pattern_num: int | list = 10
    fp: float | list = 0.005  # firing probability per pattern neuron
    mask_ratio: float | list = 0.1


@dataclasses.dataclass
class ResultInfo:
    """Result information for the experiment."""

    # num_images: int
    # masked_ratio: float
    # opt_thr_ca1: pa.int64
    real_var: pa.float64
    opt_hebbian_topk: pa.int64
    opt_err: pa.float64
    opt_var: pa.float64
    zab: pa.float64
    hdistance_x_x_masked: pa.float64
    # opt_thr_ca1_control: pa.int64
    opt_thr_ca3_control: pa.int64
    opt_err_control: pa.float64
    opt_var_control: pa.float64

    experiment_index: pa.int64 = -1


class ExperimentBTSPFeedbackTopK(auto_experiment.ExperimentInterface):
    """Experiment Interface for the BTSP Feedback Top-K experiment."""

    def __init__(self):
        self.meta_params = MetaParams(
            random_seed=0,
            precision="half",
            device="cuda",
            experiment_index=-1,
            input_size=200,
            output_dim=150,
            btsp_fq=0.005,
            btsp_topk_ratio_nfq=0.5,
            fw=0.6,
            hebbian_topk_ratio_range_mfp=(0.5, 1.5),
            pattern_num=1000,
            fp=0.005,
            mask_ratio=0.3,
        )

        # for debugging
        self.meta_params = MetaParams(
            random_seed=0,
            precision="half",
            device="cuda",
            experiment_index=-1,
            input_size=2500,
            output_dim=3900,
            btsp_fq=0.01,
            btsp_topk_ratio_nfq=[0.01, 0.05, 0.1, 0.5, 1.0],
            fw=1,
            hebbian_topk_ratio_range_mfp=(0.1, 15),
            pattern_num=[100, 200, 300, 400],
            fp=0.005,
            mask_ratio=[0.1, 0.2, 0.3],
        )

        self.data_folder = "data"
        self.experiment_name = "debug_experiment_btsp_feedback_topk"
        self.experiment_folder = logging.init_experiment_folder(
            data_folder=self.data_folder,
            experiment_name=self.experiment_name,
            timed=False,
        )

        data_schema = pa.Table.from_pydict(
            logging.dict_elements_to_tuple(dataclasses.asdict(MetaParams()))
        ).schema
        data_path = os.path.join(self.experiment_folder, "data.parquet")
        self.data_recorder = logging.ParquetTableRecorder(data_path, data_schema, 1000)

        result_schema = logging.dataclass_to_pyarrow_schema(ResultInfo)
        result_path = os.path.join(self.experiment_folder, "results.parquet")
        self.result_recorder = logging.ParquetTableRecorder(
            result_path, result_schema, 1000
        )

    def load_parameters(self):
        """Load parameters for the experiment."""
        print("Loading parameters...")
        combinations: list[MetaParams] = parameterization.recursive_iterate_dataclass(
            self.meta_params
        )
        parameters = []
        for index, combination in enumerate(combinations):
            combination.experiment_index = index
            # for debugging
            # print(logging.dict_elements_to_tuple(dataclasses.asdict(combination)))
            parameters.append(combination)
            self.data_recorder.record(logging.dict_elements_to_tuple(dataclasses.asdict(combination)))
        self.data_recorder.close()
        print("Parameters loaded.")
        return parameters

    def load_dataset(self):
        """Dataset is randomly generated, no need to load."""
        return None

    def execute_experiment_process(self, parameters: MetaParams, dataset):
        """Core experiment process."""

        num_images = parameters.pattern_num
        m = parameters.input_size
        n = parameters.output_dim
        fq = parameters.btsp_fq
        fq_half = fq / 2
        fp = parameters.fp
        p_w = parameters.fw
        device = parameters.device
        masked_ratio = parameters.mask_ratio
        btsp_topk = int(parameters.btsp_topk_ratio_nfq * n * fq)
        hebbian_topk_ratio_range_mfp = parameters.hebbian_topk_ratio_range_mfp

        # set precision
        if parameters.precision == "half":
            precison = torch.float16
        elif parameters.precision == "single":
            precison = torch.float32
        else:
            # not implemented
            raise NotImplementedError

        # set random seed
        torch.manual_seed(parameters.random_seed)
        np.random.seed(parameters.random_seed)

        # core experiment process implemented by Prof. Wu
        # Initialize weight matrices
        # print("\n\n num_images", num_images)
        num_images = int(num_images)
        X = (
            torch.Tensor(np.random.binomial(n=1, p=fp, size=(m, num_images)))
            .cuda()
            .to(precison)
            .T
        )
        plateaus = (torch.rand(num_images, n).cuda() <= fq_half).to(precison)
        # sum_W = X.T @ plateaus
        W_feed_init = 0.0
        W_back_init = 0.0
        # W_feed_init_control = 0.

        # sum_W = X.T @ plateaus
        # connection matrix for BTSP layer
        W_mask1 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
        # connection matrix for Hebbian feedback layer
        W_mask2 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
        # average firing neuron number per pattern
        
        # image_intensity = X.sum(1).mean()
        thr_ca1 = int(m * fp * p_w * 0.6)
        # thr_ca2 = int(m * fp * p_w * 0.6)
        ## select one threshold for learning
        # Method1:  fast simulation (recommended)
        # Note: Hebbian learning is perform after all BTSP learning is done
        # weight change number for each synapse throughout the learning process
        W_feed_init = X.T @ plateaus
        # incorporate the connection mask for selecting valid synapses
        W_feed_init = (W_feed_init * W_mask1) % 2
        # output of the BTSP layer
        y_sum = X @ W_feed_init
        # threshold for the BTSP layer
        spikes1 = (y_sum > thr_ca1).to(precison)
        # Hebbian learning
        W_back_init += spikes1.T @ X

        W_feed = W_feed_init * W_mask1  # this line is redundant
        W_back = W_back_init * W_mask2.T  # hebbian with synapse mask
        W_back = (W_back >= 1).to(precison)

        W_feed = W_feed_init * W_mask1
        W_back = W_back_init * W_mask2.T
        del W_mask1, W_mask2, W_back_init, W_feed_init
        torch.cuda.empty_cache()
        # training complete

        # Mask the top half of the patterns and project the results using Wf
        X_masked = X.clone()
        X_masked[:, : int(m * masked_ratio)] = 0
        reconstruct_results = []
        zab = 2 * m * (1 - X.mean()) * X.mean()
        zab = zab.item()
        err1 = (X - X_masked).abs().mean()
        ###
        # BTSP-Hebbian with feedback Top-k selection
        ###
        winner_k = btsp_topk
        input_sum_ca1 = X_masked @ W_feed
        # Get the topk indices along the dimension dim=1
        _, topk_indices = torch.topk(input_sum_ca1, k=winner_k, dim=1)

        # Create a binary mask with the same shape as input_sum_ca1, initialized to zeros
        mask = torch.zeros_like(input_sum_ca1, dtype=torch.bool)

        # Scatter 1s to the positions of the top k elements
        mask.scatter_(dim=1, index=topk_indices, value=True)
        y_ = mask.to(precison)

        # feedback
        X_projected = y_ @ W_back

        average_firing_neuron = parameters.fp * m
        # for debugging
        real_average_firing_neuron = X.sum(1).mean()
        read_var = X.sum(1).var().item()
        # print the difference ratio
        # print((average_firing_neuron - real_average_firing_neuron) / real_average_firing_neuron)
        average_firing_neuron = real_average_firing_neuron
        hebbian_topk_max = int(average_firing_neuron * hebbian_topk_ratio_range_mfp[1])
        hebbian_topk_min = int(average_firing_neuron * hebbian_topk_ratio_range_mfp[0])
        steps = int((hebbian_topk_max - hebbian_topk_min) / 100) + 1
        for hebbian_topk in range(hebbian_topk_min, hebbian_topk_max, steps):
            # apply topk selection on the projected results
            _, topk_indices = torch.topk(X_projected, k=hebbian_topk, dim=1)
            mask = torch.zeros_like(X_projected, dtype=torch.bool)
            mask.scatter_(dim=1, index=topk_indices, value=True)
            tmp = mask.float()
            err0 = (tmp - X).abs().mean()
            var = tmp.sum(1).var()
            items = [hebbian_topk, err0.item(), var.item()]
            reconstruct_results.append(items)
        reconstruct_array = np.array(reconstruct_results)

        # print(reconstruct_array,max_range,steps )
        idx_min_err = reconstruct_array[:, 1].argmin()
        opt_hebbian_topk, opt_err, opt_var = reconstruct_array[idx_min_err][:3]

        ###
        # control_model: BTSP-Hebbian without feedback Top-k selection
        ###
        reconstruct_control = []
        # can directly use the X projected by BTSP layer
        X_projected_control = X_projected
        # use grid search instead of topk selection
        if X_projected_control.max() > 200:
            max_range = 200
        else:
            max_range = max(X_projected_control.max(), 2)
        steps = int(max_range / 100) + 1
        for thr_ca3 in range(0, int(max_range), steps):
            tmp = (X_projected_control >= thr_ca3).float()
            err0 = (tmp - X).abs().mean()
            var = tmp.sum(1).var()
            items = [thr_ca3, err0.item(), var.item()]
            reconstruct_control.append(items)

        reconstruct_control_array = np.array(reconstruct_control)
        idx_min_err = reconstruct_control_array[:, 1].argmin()
        opt_thr_ca3_control, opt_err_control, opt_var_control = reconstruct_control_array[idx_min_err][
            :3
        ]

        err1 = err1.item()

        record_items = ResultInfo(
            real_var=read_var,
            opt_hebbian_topk=opt_hebbian_topk,
            opt_err=opt_err,
            opt_var=opt_var,
            zab=zab,
            hdistance_x_x_masked=err1,
            opt_thr_ca3_control=opt_thr_ca3_control,
            opt_err_control=opt_err_control,
            opt_var_control=opt_var_control,
            experiment_index=parameters.experiment_index,
        )
        record_items_dict = logging.dict_elements_to_tuple(
            dataclasses.asdict(record_items)
        )
        self.result_recorder.record(record_items_dict)
        # raw_err = err1
        # print('masking fraction {:.4f}, raw err {:.4f}, reconstruction  by btsp{:.4f} ratio {:.4f} '
        #   ' reconstruction   by control{:.4f} ratio {:.4f}'.format(masked_ratio, raw_err, opt_err,
        #                                                                opt_err / (raw_err + 1e-4),
        #                                                                opt_err_ca1_control,
        #                                                                opt_err_ca1_control / (raw_err + 1e-4),
        #                                                                ))

    def summarize_results(self):
        """Summarize results of the experiment."""
        if self.result_recorder.recording:
            self.result_recorder.close()

        # Load data/result and merge tables
        data_table = pq.read_table(os.path.join(self.experiment_folder, "data.parquet"))
        result_table = pq.read_table(os.path.join(self.experiment_folder, "results.parquet"))
        result_df = result_table.to_pandas()
        data_df = data_table.to_pandas()
        merged_df = data_df.merge(result_df, on="experiment_index")

        # merge batch data
        # for each masking ratio and pattern number, only keep 2 records:
        # 1. the record with the smallest opt_err
        # 2. the record with the smallest opt_err_control
        opt_err_min = merged_df.groupby(["mask_ratio", "pattern_num"])[
            "opt_err"
        ].idxmin()
        opt_err_control_min = merged_df.groupby(["mask_ratio", "pattern_num"])[
            "opt_err_control"
        ].idxmin()
        opt_err_min_df = merged_df.loc[opt_err_min]
        opt_err_control_min_df = merged_df.loc[opt_err_control_min]

        # start plotting
        # figure with 2 subplots
        # 1. pattern_num vs masked_ratio vs opt_err
        # 2. pattern_num vs masked_ratio vs opt_err_control
        fig, axs = plt.subplots(1, 3, figsize=(17, 5), subplot_kw={"projection": "3d"})
        # the width of the subplots are 5 inches, the extra 2 inches are for color bar
        # create a coolwarm color mapping from 0 to 1
        color_map = "coolwarm"
        norm = plt.Normalize(0, 100)
        for ax, df, title in zip(
            axs,
            [opt_err_min_df, opt_err_min_df, opt_err_control_min_df],
            ["real_var","opt_var", "opt_var_control"],
            ["Real Pattern Activation Variance", "Optimal TopK Reconstructed Pattern Activation Variance", "Optimal Grid Search Recontsructed Pattern Activation Variance Control"],
        ):
            ax.set_title(title)
            ax.set_xlabel("pattern_num")
            ax.set_ylabel("mask_ratio")
            x = df["pattern_num"]
            y = df["mask_ratio"]
            # z = df[title] / df["hdistance_x_x_masked"]
            z = df[title]
            # truncate z to 1
            # z = np.minimum(z, 1)
            ax.plot_trisurf(x, y, z, cmap=color_map, norm=norm)

        # add color bar to the right of the rightmost subplot
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[-1], orientation="vertical")

        # save the figure
        fig_path = os.path.join(self.experiment_folder, "results.png")
        plt.savefig(fig_path)

if __name__ == "__main__":
    REPEAT_NUM = 1
    experiment = auto_experiment.SimpleBatchExperiment(ExperimentBTSPFeedbackTopK(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()