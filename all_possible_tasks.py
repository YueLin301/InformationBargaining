# ========================================
# Tasks
# ========================================

"""
Bargaining:
- Ultimatum (one-shot)
    - Value setting: unbounded, bounded
    - First run proposer: coin-flip, system-assigned
    - May_meet_again_context: 
        - never meet again
        - may meet again: alternating_offer context
        - may meet again: fixed role context
- Long-term bargaining
    - Rubinstein's alternating-offer model (alternating-offer)
        - Value setting: unbounded, bounded
        - (First run proposer: coin-flip)
    - Repeated ultimatum game (fixed role)
        - Value setting: unbounded, bounded
        - First run proposer: coin-flip, system-assigned
- [Scenario]
    - pure math
    - splitting coins
    - making deals
        - seller as proposer
        - buyer as proposer


Signaling:
- Bayesian persuasion
    - One-shot Bayesian persuasion
        - (Value setting: bounded)
        - (First run proposer: system-assigned)
        - (May_meet_again_context: never meet again)
    - Long-term Bayesian persuasion (Repeated Bayesian persuasion)
        - (Value setting: bounded)
        - (First run proposer: system-assigned)
- Information bargaining
    - One-shot information bargaining
        - (Value setting: bounded)
        - (First run proposer: coin-flip); If system-assigned, then it is exactly the Bayesian persuasion task
        - May_meet_again_context: 
            - may meet again: alternating_offer context
            - may meet again: fixed role context
    - Long-term information bargaining
        - Alternating-offer model
            - (Value setting: bounded)
            - (First run proposer: coin-flip)
- [Scenario]
    - pure math
    - grading students
    - selling products
"""

"""
Task Parameters:
- task: "bargaining", "signaling"
- duration: "one_shot", "long_term"
- scenario:
    - (Bargaining)
        - pure math
        - splitting coins
        - making deals
            - seller as proposer;
                - this works only if first_run_proposer == "system_assigned"
                - if first_run_proposer == "coin_flip": agent0 is the seller, agent 1 is the buyer, and the proposer is decided by a coin-flip
            - buyer as proposer; same
    - (Signaling)
        - pure math
        - grading students
        - selling products
- value_setting: "unbounded", "bounded"; if task == "signaling", then "bounded"
- first_run_proposer: "coin_flip", "system_assigned"
- may_meet_again_context: "never_meet_again", "alternating_offer", "fixed_role"; if duration == "long_term", then None
- long_term_type: "alternating_offer", "fixed_role"; if duration == "one_shot", then None


agent_index (system): [0, 1]
scenario_type (system): 
    - ["seller", "buyer"], if task == "bargaining" and scenario begins with "making_deals_"
    - ["sender", "receiver"], if task == "signaling"
role (user): ["proposer", "responder"]
"""

# ----------------------------------------
# Other constraints
# ----------------------------------------

"""
Bargaining:
if first_run_proposer == "system_assigned"
    - (role)
        - agent 0 is the proposer, agent 1 is the responder
    - (scenario_type)
        - making_deals_seller_as_proposer: agent 0 is the seller, agent 1 is the buyer
        - making_deals_buyer_as_proposer: agent 0 is the buyer, agent 1 is the seller
        - (other scenarios): None
else (first_run_proposer == "coin_flip"):
    - (role)
        - agent 0 is the proposer with 50% chance, agent 1 is the proposer with 50% chance
    - (scenario_type)
        - making_deals_: agent 0 is the seller, agent 1 is the buyer
        - (other scenarios): None

if duration == "long_term" and long_term_type == alternating_offer
    then first_run_proposer: "coin_flip"
        

Signaling:
- (role)
    if first_run_proposer == "system_assigned"
        - agent 0 is the proposer, agent 1 is the responder
    else (first_run_proposer == "coin_flip"):
        - agent 0 is the proposer with 50% chance, agent 1 is the proposer with 50% chance
- (scenario_type)
    - agent 0 is the sender, agent 1 is the receiver

if duration == "one_shot" and first_run_proposer == "system_assigned"
    then May_meet_again_context: never meet again
if duration == "one_shot" and first_run_proposer == "coin_flip"
    then May_meet_again_context: [may meet again: alternating_offer context, may meet again: fixed role context]
if duration == "long_term" and first_run_proposer == "system_assigned"
    then long_term_type: fixed_role
if duration == "long_term" and first_run_proposer == "coin_flip"
    then long_term_type: alternating_offer

"""


def generate_all_possible_parameter_set():
    task_list = ["bargaining", "signaling"]
    duration_list = ["one_shot", "long_term"]
    scenario_dict = {
        "bargaining": ["pure_math", "splitting_coins", "making_deals_seller_as_proposer", "making_deals_buyer_as_proposer"],
        "signaling": ["pure_math", "grading_students", "selling_products"],
    }
    value_setting_dict = {
        "bargaining": ["unbounded", "bounded"],
        "signaling": ["bounded"],
    }
    first_run_proposer_list = ["coin_flip", "system_assigned"]
    may_meet_again_context_dist = {
        "one_shot": ["never_meet_again", "may_meet_again_alternating", "may_meet_again_fixed_role"],
        "long_term": [None],
    }
    long_term_type_dist = {
        "one_shot": [None],
        "long_term": ["alternating_offer", "fixed_role"],
    }

    all_possible_parameter_set = []
    for task in task_list:
        for duration in duration_list:
            for scenario in scenario_dict[task]:
                for value_setting in value_setting_dict[task]:
                    for first_run_proposer in first_run_proposer_list:
                        for may_meet_again_context in may_meet_again_context_dist[duration]:
                            for long_term_type in long_term_type_dist[duration]:
                                # other constraints
                                if task == "bargaining":
                                    if duration == "long_term" and long_term_type == "alternating_offer" and first_run_proposer != "coin_flip":
                                        continue
                                else:
                                    if duration == "one_shot":
                                        if first_run_proposer == "system_assigned" and may_meet_again_context != "never_meet_again":
                                            continue
                                        if first_run_proposer == "coin_flip" and not (
                                            may_meet_again_context in ["may_meet_again_alternating", "may_meet_again_fixed_role"]
                                        ):
                                            continue
                                    else:
                                        if first_run_proposer == "system_assigned" and long_term_type != "fixed_role":
                                            continue
                                        if first_run_proposer == "coin_flip" and long_term_type != "alternating_offer":
                                            continue
                                # append
                                current_parameter_set = (
                                    task,
                                    duration,
                                    scenario,
                                    value_setting,
                                    first_run_proposer,
                                    may_meet_again_context,
                                    long_term_type,
                                )
                                all_possible_parameter_set.append(current_parameter_set)

    return all_possible_parameter_set

def print_csv_all_possible_parameter_set(all_possible_parameter_set):
    import csv
    header = [
        "task",
        "duration",
        "scenario",
        "value_setting",
        "first_run_proposer",
        "may_meet_again_context",
        "long_term_type",
    ]

    csv_filename = "all_possible_tasks.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_possible_parameter_set)

# from pprint import pprint
# all_possible_parameter_set = generate_all_possible_parameter_set()
# pprint(all_possible_parameter_set)

print_csv_all_possible_parameter_set(generate_all_possible_parameter_set())