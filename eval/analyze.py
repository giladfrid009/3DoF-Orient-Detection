import pandas as pd

from algs.algorithm import *
from eval.eval_log import EvalLog


class LogUtils:

    @staticmethod
    def _add_params(df: pd.DataFrame, log: EvalLog) -> pd.DataFrame:
        for param, value in log.alg_params.items():
            df[param] = [value] * len(df)
        return df

    @staticmethod
    def to_dataframe(log: EvalLog, add_params: bool = False) -> pd.DataFrame:
        n_samples = len(log.eval_loss_list)

        data = {
            "alg": [*([log.alg_name] * n_samples)],
            "sample": [*range(n_samples)],
            "eval_loss": [*log.eval_loss_list],
            "ref_pos": [*log.obj_position_list],
            "pred_pos": [*log.pred_position_list],
        }

        df = pd.DataFrame(data)
        if add_params:
            df = LogUtils._add_params(df, log)

        return df

    @staticmethod
    def to_history_dataframe(log: EvalLog, add_params=False) -> pd.DataFrame:
        df_list = []

        for i, hist in enumerate(log.run_hist_list):
            data = {
                "alg": ([log.alg_name] * hist.num_epochs),
                "sample": ([i] * hist.num_epochs),
                "epoch": range(hist.num_epochs),
                "eval_loss": ([log.eval_loss_list[i]] * hist.num_epochs),
                "epoch_time": hist.epoch_time_list,
                "global_best_fit": hist.global_best_fit_list,
            }

            df = pd.DataFrame(data)
            if add_params:
                df = LogUtils._add_params(df, log)

            df_list.append(df)

        return pd.concat(df_list, axis=0, ignore_index=True)
