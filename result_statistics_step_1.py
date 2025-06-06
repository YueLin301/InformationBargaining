import os
import pandas as pd
import re

# model_name = "gpt-4o-mini"
# model_name = "gpt-4.1-mini-2025-04-14"
# model_name = "o3-2025-04-16"

# model_name = "deepseek-chat"  # V3
model_name = "deepseek-reasoner"  # R1


results_folder = f"results/{model_name}/"


def get_statistics(df):
    statistics = {}
    for column in df.columns:
        if column not in ["run_index", "file_name", "task", "file_number"]:
            statistics[f"{column}_mean"] = df[column].mean()
            statistics[f"{column}_std"] = df[column].std()
            statistics[f"{column}_min"] = df[column].min()
            statistics[f"{column}_max"] = df[column].max()
            # statistics[f"{column}_median"] = df[column].median()
            statistics[f"{column}_sample_count"] = len(df)
        elif column in ["run_index"]:
            continue
        else:
            statistics[f"{column}"] = df[column]

    return statistics


def extract_number_from_filename(filename):
    match = re.match(r"(\d+)_", filename)
    if match:
        return int(match.group(1))
    return 0


def process_files():
    result_statistics = {"bargaining": [], "signaling": []}
    for root, dirs, files in os.walk(results_folder):
        for file in files:
            if (not (file.startswith("bargaining") or file.startswith("bargaining"))) and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                stats = get_statistics(df)

                task_type = "bargaining" if "bargaining" in file else "signaling"
                stats["file_name"] = file
                stats["task"] = task_type

                stats["file_number"] = extract_number_from_filename(file)

                result_statistics[task_type].append(stats)

    for task_type, stats_list in result_statistics.items():
        result_df = pd.DataFrame(stats_list)
        result_df.sort_values(by="file_number", ascending=True, inplace=True)
        result_df.drop(columns=["file_number"], inplace=True)

        output_file = f"results/{model_name}/{task_type}_statistics_summary.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Results for '{task_type}' task saved to: {output_file}")

    print("\nOverall Summary:")
    for task_type, stats_list in result_statistics.items():
        if stats_list:
            print(f"\n{task_type.capitalize()} Task Summary:")
            summary_df = pd.DataFrame(stats_list)
            print(summary_df)


process_files()
