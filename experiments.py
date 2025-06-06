from numpy import single
from utils.Util_print import print_elapsed_time, print_separator
from openai import OpenAI
import json, sys
import random
from pprint import pprint
import csv
import math
import time
import os

# model_name = "gpt-4o-mini"
# model_name = "gpt-4.1-mini-2025-04-14"
# model_name = "o3-2025-04-16"

# model_name = "o1-mini-2024-09-12" # debug
# model_name = "gpt-4o-2024-11-20"  # debug

# model_name = "deepseek-chat"  # V3
model_name = "deepseek-reasoner"  # R1

if model_name in ["gpt-4o-mini", "gpt-4.1-mini-2025-04-14", "o3-2025-04-16", "o1-mini-2024-09-12", "gpt-4o-2024-11-20"]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif model_name in ["deepseek-chat", "deepseek-reasoner"]:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )

from prompts import (
    initialize_system_prompt,
    one_shot_make_decision_user_prompt,
    long_term_make_decision_user_prompt,
    bargaining_proposal_template,
    signaling_proposal_template,
    bargaining_long_term_memory_record,
    signaling_long_term_memory_record,
)


class agent:
    def __init__(
        self,
        task,
        duration,
        scenario,
        value_setting,
        first_run_proposer,
        may_meet_again_context,
        long_term_type,
        # ----------------------------
        model_name,
        agent_index,
        init_role,
        scenario_type,
    ):
        if task == "bargaining" and scenario.startswith("making_deals"):
            assert scenario_type in ["seller", "buyer"]
        elif task == "signaling":
            assert scenario_type in ["sender", "receiver"]
        assert init_role in ["proposer", "responder"]

        self.model_name = model_name
        self.agent_index = agent_index
        self.scenario_type = scenario_type  # if making_deals_: [seller, buyer]
        self.role = init_role  # [proposer, responder]

        self.last_payoff = None

        init_system_prompt_template = initialize_system_prompt(
            task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type
        )

        init_system_prompt = init_system_prompt_template.format(
            agent_index_prompt=f"You are the agent {agent_index}",
            scenario_type_prompt="" if scenario_type is None else f"\n- You are the {scenario_type}",
        )

        self.memory = [
            {
                "role": self.select_LLM_prompt_role(self.model_name),
                "content": init_system_prompt,
            }
        ]

    def select_LLM_prompt_role(self, model_name):
        if model_name in ["gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4-turbo-2024-04-09", "deepseek-reasoner"]:
            result = "system"
        else:
            result = "user"
        return result

    def update_memory(self, new_content):
        self.memory = self.memory + [{"role": "user", "content": new_content}]
        return

    def query_memory_then_act(self, short_memory):
        if model_name in ["deepseek-reasoner", "o1-mini-2024-09-12", "gpt-4o-2024-11-20"] and self.memory[-1]["role"] == "user":
            query = self.memory
            query[-1]["content"] += short_memory
        else:
            query = self.memory + [{"role": "user", "content": short_memory}]

        pprint(query)

        max_retries = 5
        for attempt in range(max_retries):
            output_full = client.chat.completions.create(
                model=self.model_name,
                messages=query,
                n=1,  # Generate 1 answer.
            )
            output_text = output_full.choices[0].message.content

            try:
                start = output_text.find("{\n")
                end = output_text.rfind("}") + 1
                json_text = output_text[start:end]
                output_json = json.loads(json_text.replace("\\", "\\\\"))

                if self.memory[-1]["role"] == "user" and model_name in ["deepseek-reasoner", "o1-mini-2024-09-12", "gpt-4o-2024-11-20"]:
                    self.memory.append({"role": "assistant", "content": output_text})

                return output_full, json_text, output_json
            except json.JSONDecodeError:
                # sys.stdout = sys.__stdout__
                print(f"Warning: Failed to decode JSON (Attempt {attempt + 1}/{max_retries}), retrying...")
                print("Text to Load:\n", json_text)
                time.sleep(3)

        raise ValueError("Failed to generate valid JSON after 5 attempts.")

    def flip_role(self):
        if self.role == "proposer":
            self.role = "responder"
        else:
            self.role = "proposer"
        return


def return_payoffs(task, value_setting, x, y, deal, receiver_as_proposer=None):
    # return:
    #   bargaining: proposer's payoff, responder's payoff
    #   signaling: sender's payoff, receiver's payoff

    if task == "bargaining":
        if deal:
            if value_setting == "unbounded":
                result = (x, 1 - x)
            elif value_setting == "bounded":
                result = ((1 + 2 * x) / 3, (1 - 2 * x) / 3)
        else:
            result = (0, 0)

    elif task == "signaling":
        assert receiver_as_proposer is not None
        if not receiver_as_proposer:
            x1, x2 = x  # varphi
            y1, y2 = y  # pi
            ri = 2 / 3 * ((1 - x1) * y1 + x1 * y2) + 1 / 3 * ((1 - x2) * y1 + x2 * y2)
            rj = -2 / 3 * ((1 - x1) * y1 + x1 * y2) + 1 / 3 * ((1 - x2) * y1 + x2 * y2)
            result = (ri, rj)
        else:
            if deal:
                x1, x2 = y  # varphi
                y1, y2 = 0, 1  # pi
                ri = 2 / 3 * ((1 - x1) * y1 + x1 * y2) + 1 / 3 * ((1 - x2) * y1 + x2 * y2)
                rj = -2 / 3 * ((1 - x1) * y1 + x1 * y2) + 1 / 3 * ((1 - x2) * y1 + x2 * y2)
                result = (ri, rj)
            else:
                result = (0, 0)

    return result


def check_decision(task, value_setting, x, y):
    no_problem = True
    if task == "bargaining":
        if value_setting == "unbounded":
            if x < 0 or x > 1 or not (y in [0, 1]):
                no_problem = False
        elif value_setting == "bounded":
            if x < 0 or x > 0.5 or not (y in [0, 1]):
                no_problem = False
    elif task == "signaling":
        x1, x2 = x
        y1, y2 = y
        if x1 < 0 or x1 > 1 or x2 < 0 or x2 > 1 or y1 < 0 or y1 > 1 or y2 < 0 or y2 > 1:
            no_problem = False

    return no_problem


def single_exp(model_name, task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type):

    role_list = ["proposer", "responder"]

    if first_run_proposer == "coin_flip":
        proposer_index = random.randint(0, 1)
    else:
        proposer_index = 0
    first_proposer_index = proposer_index

    if task == "bargaining":
        if first_run_proposer == "system_assigned":
            if scenario == "making_deals_seller_as_proposer":
                agent0_scenario_type = "seller"
                agent1_scenario_type = "buyer"
            elif scenario == "making_deals_buyer_as_proposer":
                agent0_scenario_type = "buyer"
                agent1_scenario_type = "seller"
            else:
                agent0_scenario_type = None
                agent1_scenario_type = None
        elif first_run_proposer == "coin_flip":
            if scenario.startswith("making_deals"):
                agent0_scenario_type = "seller"
                agent1_scenario_type = "buyer"
            else:
                agent0_scenario_type = None
                agent1_scenario_type = None
    elif task == "signaling":
        agent0_scenario_type = "sender"
        agent1_scenario_type = "receiver"

    agent0 = agent(
        task,
        duration,
        scenario,
        value_setting,
        first_run_proposer,
        may_meet_again_context,
        long_term_type,
        model_name,
        agent_index=0,
        init_role=role_list[proposer_index],
        scenario_type=agent0_scenario_type,
    )
    agent1 = agent(
        task,
        duration,
        scenario,
        value_setting,
        first_run_proposer,
        may_meet_again_context,
        long_term_type,
        model_name,
        agent_index=1,
        init_role=role_list[1 - proposer_index],
        scenario_type=agent1_scenario_type,
    )

    agents = [agent0, agent1]

    # ----------------------------------------

    deal = False
    timestep = 0

    # t_max = initialize_t_max()
    if duration == "one_shot":
        t_max = 1
        termination_prob = -1
    elif duration == "long_term":
        if long_term_type == "fixed_role":
            t_max = 5
            termination_prob = -1
        elif long_term_type == "alternating_offer":
            t_max = 10
            termination_prob = 0.1

    # ----------------------------------------

    while not deal and timestep < t_max:

        if duration == "one_shot":
            proposer_temp_query = one_shot_make_decision_user_prompt.format(role="proposer")
        elif duration == "long_term":
            proposer_temp_query = long_term_make_decision_user_prompt.format(timestep=timestep, role="proposer")

        _, proposer_output_text, proposer_output_json = agents[proposer_index].query_memory_then_act(proposer_temp_query)
        print(f"Proposer: Agent {agents[proposer_index].agent_index}:\n")
        print(proposer_output_text)

        proposer_decision = proposer_output_json["Decision"]
        if task == "bargaining":
            x = proposer_decision
            proposal = bargaining_proposal_template.format(x=x)
        elif task == "signaling":
            x1, x2 = proposer_decision

            # switch to make sure that the mining of signal 1 is recommendation
            switched = False
            if x1 > x2:
                switched = True
                x1, x2 = x2, x1
            proposal = signaling_proposal_template.format(x1=x1, x2=x2)

            if switched:
                proposal += "(signals switched)"
        print("Proposal:\n", proposal)
        print_separator(".")

        # ----------------------------------------

        if duration == "one_shot":
            responder_temp_query = one_shot_make_decision_user_prompt.format(role="responder")
        elif duration == "long_term":
            responder_temp_query = long_term_make_decision_user_prompt.format(timestep=timestep, role="responder")

        responder_temp_query = proposal + responder_temp_query
        _, responder_output_text, responder_output_json = agents[1 - proposer_index].query_memory_then_act(responder_temp_query)

        print(f"Responder: Agent {agents[1-proposer_index].agent_index}:\n")
        print(responder_output_text)
        print_separator("-")

        # ----------------------------------------

        responder_decision = responder_output_json["Decision"]
        if task == "bargaining":
            y = responder_decision
            if y == 1:
                deal = True
                responder_decision_verb = "accepted"
            else:
                responder_decision_verb = "rejected"
        elif task == "signaling":
            y1, y2 = responder_decision
            if agents[proposer_index].scenario_type == "sender":
                if math.isclose(y1, 0, abs_tol=1e-2) and math.isclose(y2, 1, abs_tol=1e-2):
                    deal = True
            elif agents[proposer_index].scenario_type == "receiver":
                if math.isclose(y1, x1, abs_tol=1e-3) and math.isclose(y2, x2, abs_tol=1e-3):
                    deal = True

        # ----------------------------------------

        no_problem = False
        if task == "bargaining":
            no_problem = check_decision(task, value_setting, x, y)
        elif task == "signaling":
            no_problem = check_decision(task, value_setting, (x1, x2), (y1, y2))

        if not no_problem:
            deal = "Error"
            break

        # ----------------------------------------

        if task == "bargaining":
            payoffs = return_payoffs(task, value_setting, x, y, deal)
            proposer_payoff = payoffs[0]
            responder_payoff = payoffs[1]
        elif task == "signaling":
            if agents[proposer_index].scenario_type == "receiver":
                receiver_as_proposer = True
            else:
                receiver_as_proposer = False

            payoffs = return_payoffs(task, value_setting, (x1, x2), (y1, y2), deal, receiver_as_proposer)

            proposer_payoff = payoffs[0] if agents[proposer_index].scenario_type == "sender" else payoffs[1]
            responder_payoff = payoffs[1] if agents[proposer_index].scenario_type == "sender" else payoffs[0]

        agents[proposer_index].last_payoff = proposer_payoff
        agents[1 - proposer_index].last_payoff = responder_payoff

        # ----------------------------------------

        if duration == "long_term":
            if task == "bargaining":
                history_record_proposer = bargaining_long_term_memory_record.format(
                    timestep=timestep,
                    proposer_index=proposer_index,
                    proposer_who="you",
                    responder_index=1 - proposer_index,
                    responder_who="your opponent",
                    x=x,
                    decision_verb=responder_decision_verb,
                    reward=proposer_payoff,
                    opponent_reward=responder_payoff,
                )
                history_record_responder = bargaining_long_term_memory_record.format(
                    timestep=timestep,
                    proposer_index=proposer_index,
                    proposer_who="your opponent",
                    responder_index=1 - proposer_index,
                    responder_who="you",
                    x=x,
                    decision_verb=responder_decision_verb,
                    reward=responder_payoff,
                    opponent_reward=proposer_payoff,
                )
            elif task == "signaling":
                history_record_proposer = signaling_long_term_memory_record.format(
                    timestep=timestep,
                    proposer_index=proposer_index,
                    proposer_who="you",
                    responder_index=1 - proposer_index,
                    responder_who="your opponent",
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y1,
                    reward=proposer_payoff,
                    opponent_reward=responder_payoff,
                )
                history_record_responder = signaling_long_term_memory_record.format(
                    timestep=timestep,
                    proposer_index=proposer_index,
                    proposer_who="your opponent",
                    responder_index=1 - proposer_index,
                    responder_who="you",
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y1,
                    reward=responder_payoff,
                    opponent_reward=proposer_payoff,
                )

            agents[proposer_index].update_memory(new_content=history_record_proposer)
            agents[1 - proposer_index].update_memory(new_content=history_record_responder)

        timestep += 1
        if deal or timestep >= t_max:
            break

        if duration == "long_term" and long_term_type == "alternating_offer":
            if random.random() < termination_prob:
                break
            proposer_index = 1 - proposer_index
            agents[0].flip_role()
            agents[1].flip_role()

    # ----------------------------------------

    if task == "bargaining":
        result = {
            "first_proposer_index": first_proposer_index,
            "last_timestep": timestep,
            "proposer_last_decision": x,
            "deal": deal,
            "agent0_payoff": agents[0].last_payoff,
            "agent1_payoff": agents[1].last_payoff,
            "last_proposer_payoff": agents[proposer_index].last_payoff,
            "last_responder_payoff": agents[1 - proposer_index].last_payoff,
        }
    elif task == "signaling":
        result = {
            "first_proposer_index": first_proposer_index,
            "last_timestep": timestep,
            "proposer_last_decision_x1": x1,
            "proposer_last_decision_x2": x2,
            "responder_last_response_y1": y1,
            "responder_last_response_y2": y2,
            "deal": deal,
            "agent0_payoff": agents[0].last_payoff,
            "agent1_payoff": agents[1].last_payoff,
            "last_proposer_payoff": agents[proposer_index].last_payoff,
            "last_responder_payoff": agents[1 - proposer_index].last_payoff,
        }
    return result


import os
from tqdm import tqdm


def exp(execute_times_k=1):
    input_csv = "task_to_execute.csv"
    all_possible_tasks_csv = "all_possible_tasks.csv"

    results_dir = os.path.join("results", model_name)
    logs_dir = os.path.join("logs", model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(all_possible_tasks_csv, "r") as file:
        all_possible_tasks = list(csv.reader(file))
        header_all_possible = all_possible_tasks[0]

    with open(input_csv, "r") as file:
        reader = csv.reader(file)
        header_task_to_execute = next(reader)

        for idx, row in enumerate(reader, start=1):
            task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type = row

            try:
                idx = (
                    all_possible_tasks.index([task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type])
                    + 1
                )
                print(
                    f"Running task idx-{idx}: {task}, {duration}, {scenario}, {value_setting}, {first_run_proposer}, {may_meet_again_context}, {long_term_type}"
                )
            except ValueError:
                print(
                    f"Warning: Setting {task}, {duration}, {scenario}, {value_setting}, {first_run_proposer}, {may_meet_again_context}, {long_term_type} not found in all_possible_tasks.csv."
                )
                continue

            base_filename = f"{idx}_task={task}_duration={duration}_scenario={scenario}_value_setting={value_setting}_first_run_proposer={first_run_proposer}_may_meet_again_context={may_meet_again_context}_long_term_type={long_term_type}"
            base_filename = base_filename.replace("/", "_").replace(" ", "_")
            result_filename = os.path.join(results_dir, f"{base_filename}.csv")
            log_filename = os.path.join(logs_dir, f"{base_filename}.log")

            results = []
            with open(log_filename, "w") as logfile:
                sys.stdout = logfile
                for i in tqdm(range(execute_times_k), desc=f"Inner Loop"):
                    print(
                        f"Executing {task}, {duration}, {scenario}, {value_setting}, {first_run_proposer}, {may_meet_again_context}, {long_term_type}, run {i+1}"
                    )
                    result = single_exp(
                        model_name, task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type
                    )
                    results.append(result)

            sys.stdout = sys.__stdout__

            all_keys = set()
            for res in results:
                all_keys.update(res.keys())
            all_keys = sorted(all_keys)

            with open(result_filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["run_index"] + all_keys)
                for i, res in enumerate(results):
                    csv_writer.writerow([i + 1] + [res.get(key, "") for key in all_keys])

    print("All tasks executed successfully!")


# ========================================


if __name__ == "__main__":

    # exp()
    exp(execute_times_k=12)
