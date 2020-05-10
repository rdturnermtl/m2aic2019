from data_io import read_as_df
import os
from data_manager import DataManager
import numpy as np

from bayesmark.data import _csv_loader
from bayesmark.stats import robust_standardize


task_map = {'binary.classification': "clf",
            "multiclass.classification": "clf",
            "multilabel.classification": "clf",
            "regression": "reg"}


skip = ('alexis', 'dexter', "flora", "grigoris", "marco", 'newsgroups', 'tania', 'wallis')

q_level = 0.86
clip_x = 100

# TODO rename input folder
data_dir = '~/bo_comp_data/auto_ml_data'
data_dir = os.path.expanduser(data_dir)
print(data_dir)

export_dir = '~/bo_comp_data/auto_ml_data_export'
export_dir = os.path.expanduser(export_dir)
os.makedirs(export_dir, exist_ok=True)

datasets = sorted(os.listdir(data_dir))
print(datasets)

type_ = "train"

success = []
fail = []
for data_name in datasets:
    print("-" * 20)

    if data_name.startswith(".") or (data_name in skip):
        print(f"skipping {data_name}")
    else:
        print(f"loading {data_name}")
        basename = os.path.join(data_dir, data_name, data_name)
        try:
            dm = DataManager(basename)
            solution_file = basename + '_' + type_ + '.solution'
            task = dm.getTypeProblem(solution_file)

            df = read_as_df(basename, type=type_, task=task)
        except RuntimeError as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            fail.append(data_name)
        else:
            all_missing = df.isnull().any(axis=0)
            df = df.loc[:, ~all_missing]

            assert df.columns[-1] == "target"
            assert np.all(df.index.values == np.arange(len(df))), "Assuming simple index."

            assert not df["target"].isnull().any()

            cat_cols = sorted(cc for cc in df.columns if cc.endswith("_cat") or df[cc].dtype.kind == "O")
            print(cat_cols)
            assert len(cat_cols) == 0

            assert all(dd in (np.int_, np.float_) for dd in df.dtypes)

            df = df.fillna(df.median())
            assert not df.isnull().any().any()
            assert np.all(np.isfinite(df.values))

            assert task in task_map
            fname = "%s-%s.csv" % (task_map[task], data_name)
            export_name = os.path.join(export_dir, fname)

            print(f"writing {data_name}")
            df.to_csv(export_name, na_rep='', header=True, index=False)

            dataset_name = "%s-%s" % (task_map[task], data_name)
            print(dataset_name)
            # TODO check problem_type_bm
            data_bm, target_bm, problem_type_bm = _csv_loader(dataset_name, return_X_y=True, data_root=export_dir, clip_x=clip_x)

            target = df.pop("target").values

            data = df.values
            data = robust_standardize(data, q_level=q_level)
            data = np.clip(data, -clip_x, clip_x)

            assert np.allclose(data, data_bm)
            # TODO
            # assert np.allclose(target, target_bm)

            success.append(data_name)

print("success:")
print("\n".join(success))
print("fail:")
print("\n".join(fail))
