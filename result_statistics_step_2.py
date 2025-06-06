import pandas as pd

# model_name = "gpt-4o-mini"
# model_name = "gpt-4.1-mini-2025-04-14"
# model_name = "o3-2025-04-16"

# model_name = "deepseek-chat"  # V3
model_name = "deepseek-reasoner"  # R1


def process_csv(file_path, columns_to_keep, rename_columns=False):
    df = pd.read_csv(file_path)
    df = df[columns_to_keep]

    for column in df.columns:
        if column.endswith("_mean"):
            std_column = column.replace("_mean", "_std")
            if std_column in df.columns:
                df[column] = df.apply(lambda row: f"${row[column]:.2f} \\pm {row[std_column]:.2f}$", axis=1)
                df = df.drop(std_column, axis=1)

        elif column.endswith("_min"):
            max_column = column.replace("_min", "_max")
            if max_column in df.columns:
                df[column] = df.apply(lambda row: f"${row[column]:.2f}, {row[max_column]:.2f}$", axis=1)
                df = df.drop(max_column, axis=1)

    if rename_columns:
        df.columns = [col.replace("_", " ") for col in df.columns]

    return df


bargaining_columns = [
    "agent0_payoff_mean",
    "agent0_payoff_std",
    "agent0_payoff_min",
    "agent0_payoff_max",
    "agent1_payoff_mean",
    "agent1_payoff_std",
    "agent1_payoff_min",
    "agent1_payoff_max",
    "deal_mean",
    "deal_std",
    "last_timestep_mean",
    "last_timestep_std",
    "last_timestep_min",
    "last_timestep_max",
    "last_proposer_payoff_mean",
    "last_proposer_payoff_std",
    "last_proposer_payoff_min",
    "last_proposer_payoff_max",
    "last_responder_payoff_mean",
    "last_responder_payoff_std",
    "last_responder_payoff_min",
    "last_responder_payoff_max",
    "proposer_last_decision_mean",
    "proposer_last_decision_std",
    "proposer_last_decision_min",
    "proposer_last_decision_max",
    "file_name",
    "task",
]

signaling_columns = [
    "agent0_payoff_mean",
    "agent0_payoff_std",
    "agent0_payoff_min",
    "agent0_payoff_max",
    "agent1_payoff_mean",
    "agent1_payoff_std",
    "agent1_payoff_min",
    "agent1_payoff_max",
    "deal_mean",
    "deal_std",
    "last_timestep_mean",
    "last_timestep_std",
    "last_timestep_min",
    "last_timestep_max",
    "last_proposer_payoff_mean",
    "last_proposer_payoff_std",
    "last_proposer_payoff_min",
    "last_proposer_payoff_max",
    "last_responder_payoff_mean",
    "last_responder_payoff_std",
    "last_responder_payoff_min",
    "last_responder_payoff_max",
    "proposer_last_decision_x1_mean",
    "proposer_last_decision_x1_std",
    "proposer_last_decision_x1_min",
    "proposer_last_decision_x1_max",
    "proposer_last_decision_x2_mean",
    "proposer_last_decision_x2_std",
    "proposer_last_decision_x2_min",
    "proposer_last_decision_x2_max",
    "responder_last_response_y1_mean",
    "responder_last_response_y1_std",
    "responder_last_response_y1_min",
    "responder_last_response_y1_max",
    "responder_last_response_y2_mean",
    "responder_last_response_y2_std",
    "responder_last_response_y2_min",
    "responder_last_response_y2_max",
    "file_name",
    "task",
]


bargaining_file = f"results/{model_name}/bargaining_statistics_summary.csv"
bargaining_df = process_csv(bargaining_file, bargaining_columns, rename_columns=True)
bargaining_df.to_csv(f"results/{model_name}/bargaining_final_result.csv", index=False)

signaling_file = f"results/{model_name}/signaling_statistics_summary.csv"
signaling_df = process_csv(signaling_file, signaling_columns, rename_columns=True)
signaling_df.to_csv(f"results/{model_name}/signaling_final_result.csv", index=False)
