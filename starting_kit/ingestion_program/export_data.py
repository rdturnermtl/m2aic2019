import warnings
from data_io import read_as_df
import os
from data_manager import DataManager
import numpy as np

from bayesmark.data import _csv_loader
from bayesmark.stats import robust_standardize


input_dir = '~/bo_comp_data/auto_ml_data'
output_dir = '~/bo_comp_data/auto_ml_data_export'
type_ = "train"
skip = ()

q_level = 0.86
clip_x = 100

task_map = {'binary.classification': "clf",
            "multiclass.classification": "clf",
            "multilabel.classification": "clf",
            "regression": "reg"}

input_dir = os.path.expanduser(input_dir)
print(input_dir)
output_dir = os.path.expanduser(output_dir)
print(output_dir)

os.makedirs(output_dir, exist_ok=True)

datasets = sorted(os.listdir(input_dir))
print(datasets)

for data_name in datasets:
    print("-" * 20)

    if data_name.startswith(".") or (data_name in skip):
        print(f"skipping {data_name}")
    else:
        print(f"loading {data_name}")
        basename = os.path.join(input_dir, data_name, data_name)

        dm = DataManager(basename)
        solution_file = basename + '_' + type_ + '.solution'
        task = dm.getTypeProblem(solution_file)

        format_ = dm.getFormatData(basename + '_train.data')
        print(f"format: {format_}")

        if format_ in ("sparse", "sparse_binary"):
            print(f"skipping {data_name}")
            continue
        assert format_ == "dense"

        df = read_as_df(basename, type=type_, task=task, max_rows=50)

        # Validate cols and index
        assert len(set(df.columns.tolist())) == df.shape[1]
        assert df.columns[-1] == "target"
        assert np.all(df.index.values == np.arange(len(df))), "Assuming simple index."

        # Check dtypes
        assert all(dd in (np.int_, np.float_) for dd in df.dtypes)
        assert len(sorted(cc for cc in df.columns if cc.endswith("_cat") or df[cc].dtype.kind == "O")) == 0
        dtype_dict = {cc: np.float_ for cc in df.columns if cc != "target"}
        df = df.astype(dtype_dict)

        # Remove all missing col
        all_missing = df.isnull().any(axis=0)
        if np.any(all_missing):
            warnings.warn("%d/%d columns all missing in %s." % (np.sum(all_missing), df.shape[1], data_name))
        df = df.loc[:, ~all_missing]
        assert not df["target"].isnull().any()

        # Fill in missing values, then nothing missing
        df = df.fillna(df.median())
        assert not df.isnull().any().any()
        assert np.all(np.isfinite(df.values))

        # Build output path
        assert task in task_map
        fname = "%s-%s.csv" % (task_map[task], data_name)
        export_name = os.path.join(output_dir, fname)

        # Final checks, again
        assert len(set(df.columns.tolist())) == df.shape[1]
        assert df.columns[-1] == "target"
        assert np.all(df.index.values == np.arange(len(df))), "Assuming simple index."
        assert all(df[cc].dtype == np.float_ for cc in df if cc != "target")
        assert df["target"].dtype == (np.int_ if task_map[task] == "clf" else np.float_)
        assert np.all(np.isfinite(df.values))

        # Export
        print(f"writing {data_name}")
        df.to_csv(export_name, na_rep='', header=True, index=False)

        # Load in from Bayesmark for round-trip test
        dataset_name = "%s-%s" % (task_map[task], data_name)
        print(dataset_name)
        data_bm, target_bm, problem_type_bm = _csv_loader(dataset_name, return_X_y=True, data_root=output_dir, clip_x=clip_x)
        print(str(problem_type_bm))

        # Do same pre-proc on original data
        target = df.pop("target").values
        data = df.values
        data = robust_standardize(data, q_level=q_level)
        data = np.clip(data, -clip_x, clip_x)
        if task_map[task] == "reg":
            target = robust_standardize(target, q_level=q_level)

        # Do the round-trip test
        assert np.allclose(data, data_bm)
        assert np.allclose(target, target_bm)
        assert str(problem_type_bm) == "ProblemType.%s" % task_map[task]

print("done")
