from data_io import read_as_df
import os
from data_manager import DataManager

# TODO rename input folder
data_dir = '~/bo_comp_data/auto_ml_data'
data_dir = os.path.expanduser(data_dir)
print(data_dir)

export_dir = '~/bo_comp_data/auto_ml_data_export'
export_dir = os.path.expanduser(export_dir)
os.makedirs(export_dir, exist_ok=True)

datasets = sorted(os.listdir(data_dir))

type_ = "train"

success = []
fail = []
for data_name in datasets:
    print("-" * 20)

    export_name = os.path.join(export_dir, data_name + ".csv")

    if data_name.startswith("."):
        print(f"skipping {data_name}")
    elif os.path.isfile(export_name):
        print(f"skipping {data_name}")
    else:
        print(f"loading {data_name}")
        basename = os.path.join(data_dir, data_name, data_name)
        try:
            dm = DataManager(basename)
            solution_file = basename + '_' + type_ + '.solution'
            task = dm.getTypeProblem(solution_file)

            data = read_as_df(basename, type=type_, task=task)
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            fail.append(data_name)
        else:
            # TODO cleanup:
            #    median interpolation for missing, assert all finite
            #    dtype all int or float
            #    cat_cols matches nothing
            #    make sure label put in last place
            print(f"writing {data_name}")
            data.to_csv(export_name, na_rep='', header=True, index=False)
            success.append(data_name)

print("success:")
print("\n".join(success))
print("fail:")
print("\n".join(fail))
