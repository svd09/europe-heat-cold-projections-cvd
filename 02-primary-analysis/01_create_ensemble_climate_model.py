import pandas as pd
import numpy as np
from pathlib import Path
import gc

data_dir = Path("data/raw_cmip6_unprocessed")
ssp245_models = [
    "CNRM-ESM2-1_SSP245_2020-80.csv",
    "GFDL-ESM4_SSP245_2020-80.csv",
    "MIROC6_SSP245_2020-80.csv",
    "NorESM2-MM_SSP245_2020-80.csv",
    "UKESM1-0-LL_SSP245_2020-80.csv",
]
ssp585_models = [
    "CNRM-ESM2-1_SSP585_2020-80.csv",
    "GFDL-ESM4_SSP585_2020-80.csv",
    "MIROC6_SSP585_2020-80.csv",
    "NorESM2-MM_SSP585_2020-80.csv",
    "UKESM1-0-LL_SSP585_2020-80.csv",
]


def create_ensemble_long_format(
    model_files, scenario_name, data_directory, chunksize=500000
):
    file_paths = []
    for i, model_file in enumerate(model_files, 1):
        file_path = data_directory / model_file
        if not file_path.exists():
            continue
        file_path.stat().st_size / 1024**3
        file_paths.append(file_path)
    if len(file_paths) < len(model_files):
        pass
    if not file_paths:
        raise ValueError(f"No valid data files found for {scenario_name}")
    output_file = data_directory / f"ENSEMBLE_{scenario_name}_2020-80.csv"
    chunk_num = 0
    total_rows = 0
    first_chunk = True
    iterators = [pd.read_csv(fp, chunksize=chunksize) for fp in file_paths]
    try:
        while True:
            chunk_num += 1
            chunks = []
            try:
                for iterator in iterators:
                    chunk = next(iterator)
                    chunks.append(chunk)
            except StopIteration:
                break
            if not chunks:
                break
            if chunk_num == 1:
                for i, chunk in enumerate(chunks[1:], 2):
                    if chunk.shape != chunks[0].shape:
                        pass
            ensemble_chunk = chunks[0][
                ["grid_id", "lon_idx", "lat_idx", "center_lon", "center_lat", "date"]
            ].copy()
            mean_values = np.stack([chunk["mean"].values for chunk in chunks])
            ensemble_chunk["mean"] = np.mean(mean_values, axis=0)
            if chunk_num == 1:
                if (
                    ensemble_chunk["mean"].min() < 200
                    or ensemble_chunk["mean"].max() > 350
                ):
                    pass
                else:
                    pass
            if first_chunk:
                ensemble_chunk.to_csv(output_file, index=False, mode="w")
                first_chunk = False
            else:
                ensemble_chunk.to_csv(output_file, index=False, mode="a", header=False)
            total_rows += len(ensemble_chunk)
            if chunk_num % 10 == 0:
                pass
            del chunks, mean_values, ensemble_chunk
            gc.collect()
    except Exception:
        raise
    finally:
        for iterator in iterators:
            try:
                iterator.close()
            except Exception:
                pass
    return str(output_file)


def main():
    if not data_dir.exists():
        return
    CHUNKSIZE = 500000
    try:
        create_ensemble_long_format(ssp245_models, "SSP245", data_dir, CHUNKSIZE)
    except Exception:
        import traceback

        traceback.print_exc()
    try:
        create_ensemble_long_format(ssp585_models, "SSP585", data_dir, CHUNKSIZE)
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
