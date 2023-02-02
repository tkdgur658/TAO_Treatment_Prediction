import os

import pandas as pd

csv_columns = [
    'Experiment Date',
    'Test Date',
    'Iteration',
    'Model Name',
    'Loss Function',
    'Loss',
    'AUROC',
    'AUPRC',
    'Accuracy',
    'Macro F1',
    'Sensitivity',
    'Specificity',
    'Precision',
    'Params',
    'FLOPs',
    'Best Epoch',
    'Batch Size',
]

class ExperimantResult():
    def __init__(
        self,
        experiment_date: str=None,
        test_date: str=None,
        iteration: int=None,
        model_name: str=None,
        loss_function: str=None,
        loss: float=None,
        auroc=None,
        auprc=None,
        accuracy=None,
        macro_f1=None,
        sensitivity=None,
        specificity=None,
        precision=None,
        params=None,
        flops=None,
        best_epoch=None,
        batch_size=None,
    ):
        self.experiment_date = experiment_date
        self.test_date = test_date
        self.iteration = iteration
        self.model_name = model_name
        self.loss_function = loss_function
        self.loss = loss
        self.auroc = auroc
        self.auprc = auprc
        self.accuracy = accuracy
        self.macro_f1 = macro_f1
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.precision = precision
        self.params = params
        self.flops = flops
        self.best_epoch = best_epoch
        self.batch_size = batch_size

def csv_logger(file: str, data: ExperimantResult):

    if not os.path.exists(file):
        df = pd.DataFrame(columns=csv_columns)
    else:
        df = pd.read_csv(file)

    df_add = pd.DataFrame(columns=csv_columns)

    for col in csv_columns:
        data_attr_name = col.lower().replace(' ', '_')
        data_attr = data.__getattribute__(data_attr_name)

        if data_attr is not None:
            if isinstance(data_attr, float):
                data_attr = f'{data_attr:.4f}'
            df_add.at[0, col] = data_attr

    df = pd.concat([df, df_add])

    df.to_csv(file, index=False)
