# ========================================
# init system prompt framework
# ========================================

init_system_prompt_framework = """## Self-Awareness

You are a self-interested rational player. 
- "Self-interested" means that you only care your own utilitarian payoffs, without necessarily considering the welfare of others. Even though sometimes you design your strategy depending on the other party's utility function, your ultimate goal is still to optimize your own expected payoffs.
- "Rational" means that you will always choose the strategy that brings you a higher expected payoff. That is, given any two strategies A and B, if strategy A provides a higher expected payoff than strategy B, you will always choose strategy A over strategy B. Even if A brings only a small improvement.
- Therefore, when making decisions, you need to compare and ensure that this strategy brings a higher expected payoff than any other strategy you could choose.

## Task Description

Apart from you, there is another self-interested rational player, and you two are going to play a game. One player acts as the {role_type1} while the other player acts as the {role_type2}. Both parties strive to maximize their own rewards.

### Task Scenario

{task_scenario_description}

### Reward Function

{reward_function_description}

### Task Procedure

The procedure of this task is as follows:

{task_procedure_description}

{may_meet_again_type_or_long_term_termination_t_description}

### Format

#### If You Are the Proposer
Format the output in JSON according to the following template:
{proposer_decision_format}

#### If You Are the Responder
Format the output in JSON according to the following template:
{responder_decision_format}

Please STRICTLY adhere to the JSON templates when outputting, and do not output anything else. Please think step by step, and then make a decision based on all the information you know. Remember that you and your opponents are both self-interested rational players. Be aware of the consequences of your decisions. Your analysis and decisions should remain logically CONSISTENT.

## Identity

- {agent_index_prompt}{scenario_type_prompt}"""


one_shot_make_decision_user_prompt = """You are the {role}. Please make a decision based on all the information you know."""

long_term_make_decision_user_prompt = (
    """The current timestep is {timestep} and you are the {role}. Please make a decision based on all the information you know."""
)


bargaining_proposal_template = """Now the proposer decides that x={x}. """

signaling_proposal_template = """Now the proposer decides that x1={x1} and x2={x2}. """


# long_term_memory_record = """At timestep {timestep}, you are the {role1} and your opponent is {role2}. The proposer proposes an offer as {offer}, """

bargaining_long_term_memory_record = """At timestep {timestep}: the proposer was agent {proposer_index} ({proposer_who}) and the responder was agent {responder_index} ({responder_who}). 
- At that time, agent {proposer_index} ({proposer_who}) proposed to specify $x$ as {x}. 
- And agent {responder_index} ({responder_who}) {decision_verb} the proposal.
- Thus, your reward is {reward} and your opponent's reward is {opponent_reward}."""

signaling_long_term_memory_record = """At timestep {timestep}: the proposer was agent {proposer_index} ({proposer_who}) and the responder was agent {responder_index} ({responder_who}). 
- At that time, agent {proposer_index} ({proposer_who}) proposed to specify $(x1, x2)$ as ({x1}, {x2}). 
- And agent {responder_index} ({responder_who}) decides to specify $(y1, y2)$ as ({y1}, {y2}).
- Thus, your expected reward is {reward} and your opponent's expected reward is {opponent_reward}."""



import textwrap
import string


def initialize_system_prompt(task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type):

    # role_type

    if task == "bargaining":
        role_type1 = "proposer"
        role_type2 = "responder"

    elif task == "signaling":
        role_type1 = "sender"
        role_type2 = "receiver"

    # ========================================
    # task_scenario_description

    # if bounded:
    #   0 <= x <= 0.5 and r^i = (1+2*x)/3 and r^j = (1-2*x)/3
    # is equivalent to:
    #   0.5 <= x <= 1 and r^i = x and r^j = 1-x
    # here we choose the first one

    if task == "bargaining":
        if value_setting == "unbounded":
            bargaining_max_x = "1"
        else:
            bargaining_max_x = "0.5"

    if scenario == "pure_math":
        if task == "bargaining":
            task_scenario_description = textwrap.dedent(
                f"""\
                This is a purely mathematical problem, with no real-world context necessary. Our focus is solely on the abstract properties of numbers and structures.
                - The proposer decides an amount $x$ (a real number such that $0 <= x <= {bargaining_max_x}$), meaning that it proposes to keep $x$ for itself and left ${bargaining_max_x}-x$ for the responder.
                - The responder decides whether to accept it. The responder's decision is represented by $y$, where $y = 0$ or $y = 1$."""
            )

        elif task == "signaling":
            task_scenario_description = textwrap.dedent(
                """\
                This is a purely mathematical problem, with no real-world context necessary. Our focus is solely on the abstract properties of numbers and structures.
                - Environmental state: 0 or 1
                - Prior state distribution: $mu_0(0) = 2/3$ and $mu_0(1) = 1/3$
                - The sender's signal: 0 or 1
                - The receiver's action: 0 or 1
                - The sender is to decide a signaling scheme $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
                - The receiver is to decide an action rule $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$."""
            )

    elif scenario == "splitting_coins":
        task_scenario_description = textwrap.dedent(
            f"""\
            You two are to divide a certain amount of money.
            - The proposer will suggest how to split the money as a real number $x$, where $0 <= x <= {bargaining_max_x}$, specifying the percentage it proposes to leave for itself.
            - The responder then decides whether to accept this offer (to get the remaining money ${bargaining_max_x}-x$). The decision of the responder is represented by $y$, where $y = 0$ or $y = 1$."""
        )

    elif scenario.startswith("making_deals"):
        task_scenario_description = textwrap.dedent(
            f"""\
            You two are negotiating a deal on a product, specifically its price.
            - The proposer will suggest a price as a real number $x$, where $0 <= x <= {bargaining_max_x}$.
            - The responder then decides whether to accept the offer at this price. If it accepts, it will get ${bargaining_max_x}-x$. The decision of the responder is represented by $y$, where $y = 0$ or $y = 1$."""
        )

    elif scenario == "grading_students":
        task_scenario_description = textwrap.dedent(
            """\
            - Background: Some recent graduates are entering the job market.
            - State and prior state distribution: Of these graduates, one third are excellent ($s=1, mu_0(1)=1/3$), and two thirds are weak ($s=0, mu_0(0)=2/3$).
            - The sender and the signal space: A professor can directly see the students' qualities. The professor can grade students as 0 (not recommend) or 1 (recommend) and then report the grades as signals to the HR.
            - The receiver and its action space: An HR can decide whether to hire based on the grades given by the professor. No to hire: 0; Hire: 1.
            - The sender is to decide a signaling scheme $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
            - The receiver is to decide an action rule $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$."""
        )

    elif scenario == "selling_products":
        task_scenario_description = textwrap.dedent(
            """\
            - Background: In a market, a variety of products are available for sale.
            - State and prior state distribution: Of these products, one third are of good quality ($s=1, mu_0(1)=1/3$), and two thirds are of bad quality ($s=0, mu_0(0)=2/3$).
            - The sender and the signal space: A seller can directly see the quality of their products. The seller can mark products as 0 (not recommend) or 1 (recommend) and then report them as signals to the buyer.
            - The receiver and its action space: A buyer decides whether to purchase based on the signals provided by the seller. No to buy: 0; Buy: 1.
            - The sender is to decide a signaling scheme $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
            - The receiver is to decide an action rule $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$."""
        )

    # ========================================
    # reward_function_description

    if value_setting == "unbounded":
        reward_expression_i = "x"
        reward_expression_j = "(1-x)"
    else:
        reward_expression_i = "(1+2*x)/3"
        reward_expression_j = "(1-2*x)/3"

    if task == "bargaining":
        reward_function_description = textwrap.dedent(
            f"""\
            - If the responder accepts the offer ($y = 1$), the proposer receives a payoff of ${reward_expression_i}$, while the responder gets ${reward_expression_j}$.
            - If the responder rejects the offer ($y = 0$), both of them get nothing."""
        )

    elif task == "signaling":
        if scenario == "pure_math":
            reward_function_description = textwrap.dedent(
                """\
                - If state=0 and action=0, the sender gets 0 (r^i(s=0, a=0)=0) and the receiver gets 0 (r^j(s=0, a=0)=0)
                - If state=0 and action=1, the sender gets 1 (r^i(s=0, a=1)=1) and the receiver gets -1 (r^j(s=0, a=1)=-1)
                - If state=1 and action=0, the sender gets 0 (r^i(s=1, a=0)=0) and the receiver gets 0 (r^j(s=1, a=0)=0)
                - If state=1 and action=1, the sender gets 1 (r^i(s=1, a=1)=1) and the receiver gets 1 (r^j(s=1, a=1)=1)
                
                Let x1, x2, y1 and y2 represent
                - $varphi(sigma=1 | s=0)$ (the probability of the sender sending signal 1 when the state is 0),
                - $varphi(sigma=1 | s=1)$ (the probability of the sender sending signal 1 when the state is 1),
                - $pi(a=1 | sigma=0)$ (the probability of the receiver taking action 1 when the signal is 0), and
                - $pi(a=1 | sigma=1)$ (the probability of the receiver taking action 1 when the signal is 1), respectively
                Then,
                - The sender's expected payoff is:
                    E(r^i) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^i(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^i(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^i(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^i(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^i(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^i(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^i(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^i(s=1, a=1)

                - The receiver's expected payoff is: 
                    E(r^j) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^j(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^j(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^j(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^j(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^j(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^j(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^j(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^j(s=1, a=1)"""
            )

        elif scenario == "grading_students":
            reward_function_description = textwrap.dedent(
                """\
                - The professor's goal is to maximize the number of students hired, as each hire yields a reward.
                    - If state=0 and action=1, the sender (the professor) gets 1 (r^i(s=0, a=1)=1)
                    - If state=1 and action=1, the sender (the professor) gets 1 (r^i(s=1, a=1)=1)
                - Conversely, the HR aims to hire as many excellent students as possible, gaining a reward for each excellent student hired and incurring a penalty for each weak student hired. 
                    - If state=0 and action=1, the receiver (the HR) gets -1 (r^j(s=0, a=1)=-1)
                    - If state=1 and action=1, the receiver (the HR) gets 1 (r^j(s=1, a=1)=1)
                - There is no reward or penalty for both players if a student is not hired.
                    - If state=0 and action=0, the sender (the professor) gets 0 and the receiver (the HR) gets 0 (r^i(s=0, a=0)=0 and r^j(s=0, a=0)=0)
                    - If state=1 and action=0, the sender (the professor) gets 0 and the receiver (the HR) gets 0 (r^i(s=1, a=0)=0 and r^j(s=1, a=0)=0)
                    
                Let x1, x2, y1 and y2 represent
                - $varphi(sigma=1 | s=0)$ (the probability of the sender sending signal 1 when the state is 0),
                - $varphi(sigma=1 | s=1)$ (the probability of the sender sending signal 1 when the state is 1),
                - $pi(a=1 | sigma=0)$ (the probability of the receiver taking action 1 when the signal is 0), and
                - $pi(a=1 | sigma=1)$ (the probability of the receiver taking action 1 when the signal is 1), respectively
                Then,
                - The sender's expected payoff is:
                    E(r^i) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^i(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^i(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^i(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^i(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^i(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^i(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^i(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^i(s=1, a=1)

                - The receiver's expected payoff is: 
                    E(r^j) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^j(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^j(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^j(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^j(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^j(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^j(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^j(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^j(s=1, a=1)"""
            )

        elif scenario == "selling_products":
            reward_function_description = textwrap.dedent(
                """\
                - The seller's goal is to maximize the number of products sold, as each sale yields a reward.
                    - If state=0 and action=1, the sender (the seller) gets 1 (r^i(s=0, a=1)=1)
                    - If state=1 and action=1, the sender (the seller) gets 1 (r^i(s=1, a=1)=1)
                - Conversely, the buyer aims to purchase as many good products as possible, gaining a reward for each good product purchased and incurring a penalty for each bad product purchased.
                    - If state=0 and action=1, the receiver (the buyer) gets -1 (r^j(s=0, a=1)=-1)
                    - If state=1 and action=1, the receiver (the buyer) gets 1 (r^j(s=1, a=1)=1)
                - There is no reward or penalty for both players if a product is not purchased.
                    - If state=0 and action=0, the sender (the seller) gets 0 and the receiver (the buyer) gets 0 (r^i(s=0, a=0)=0 and r^j(s=0, a=0)=0)
                    - If state=1 and action=0, the sender (the seller) gets 0 and the receiver (the buyer) gets 0 (r^i(s=1, a=0)=0 and r^j(s=1, a=0)=0)
                    
                Let x1, x2, y1 and y2 represent
                - $varphi(sigma=1 | s=0)$ (the probability of the sender sending signal 1 when the state is 0),
                - $varphi(sigma=1 | s=1)$ (the probability of the sender sending signal 1 when the state is 1),
                - $pi(a=1 | sigma=0)$ (the probability of the receiver taking action 1 when the signal is 0), and
                - $pi(a=1 | sigma=1)$ (the probability of the receiver taking action 1 when the signal is 1), respectively
                Then,
                - The sender's expected payoff is:
                    E(r^i) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^i(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^i(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^i(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^i(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^i(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^i(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^i(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^i(s=1, a=1)

                - The receiver's expected payoff is: 
                    E(r^j) = 
                        mu_0(s=0) * (1-x1) * (1-y1) * r^j(s=0, a=0)
                        + mu_0(s=0) * (1-x1) * y1 * r^j(s=0, a=1)
                        + mu_0(s=0) * x1 * (1-y2) * r^j(s=0, a=0)
                        + mu_0(s=0) * x1 * y2 * r^j(s=0, a=1)
                        + mu_0(s=1) * (1-x2) * (1-y1) * r^j(s=1, a=0)
                        + mu_0(s=1) * (1-x2) * y1 * r^j(s=1, a=1)
                        + mu_0(s=1) * x2 * (1-y2) * r^j(s=1, a=0)
                        + mu_0(s=1) * x2 * y2 * r^j(s=1, a=1)"""
            )

    # ========================================
    # task_procedure_description

    if first_run_proposer == "coin_flip":
        first_run_proposer_description = "a coin flip"
    elif first_run_proposer == "system_assigned":
        first_run_proposer_description = "the system, inherently"

    if task == "bargaining":
        if duration == "one_shot":
            task_procedure_description = textwrap.dedent(
                f"""\
                1. Who to be the proposer (in the first run) is determined by {first_run_proposer_description}.
                2. The proposer makes a decision by specifying $x$, meaning that it decides to keep ${reward_expression_i}$ for itself.
                3. The responder decides whether to accept or reject the offer (${reward_expression_j}$) by specifying $y$.
                4. Each agent gets a reward based on the decisions $x$ and $y$."""
            )

        elif duration == "long_term":
            if long_term_type == "fixed_role":
                task_procedure_description = textwrap.dedent(
                    f"""\
                    1. Who to be the proposer (in the first run) is determined by {first_run_proposer_description}.
                    2. The following process continues until one of two conditions is met: either a consensus is reached ($y = 1$) or the game ends due to a timeout:
                        3. The proposer makes a decision by specifying $x$, meaning that it decides to keep ${reward_expression_i}$ for itself.
                        4. The responder decides whether to accept or reject the offer (${reward_expression_j}$) by specifying $y$.
                    5. If a consensus is reached, each agent receives a reward based on the final offer $x$. If the game ends without a consensus, both players receive nothing."""
                )

            elif long_term_type == "alternating_offer":
                task_procedure_description = textwrap.dedent(
                    f"""\
                    1. Who to be the proposer (in the first run) is determined by {first_run_proposer_description}.
                    2. The following process continues until one of two conditions is met: either a consensus is reached ($y = 1$) or the game ends due to a timeout:
                        3. The proposer makes a decision by specifying $x$, meaning that it decides to keep ${reward_expression_i}$ for itself.
                        4. The responder decides whether to accept or reject the offer (${reward_expression_j}$) by specifying $y$.
                        5. If the responder rejects ($y = 0$), the two agents switch roles: the current responder becomes the proposer, and the current proposer becomes the responder.
                    6. If a consensus is reached, each agent gets a reward based on the final offer $x$. If the game ends without a consensus, both players receive nothing."""
                )

    elif task == "signaling":
        if duration == "one_shot":
            if first_run_proposer == "system_assigned":
                task_procedure_description = textwrap.dedent(
                    """\
                    1. The sender determines a signaling scheme $varphi$ and commits it to the receiver.
                    2. The receiver decides an action rule: 
                        - $pi_0$: The receiver ignores the sender's signals and chooses the best response to the prior belief at each time in the sample phase.
                        - $pi_1$: The receiver calculates its posterior belief (using prior belief, the sender's signaling scheme, and every sent signal in the sample phase), and chooses the best response to the posterior belief.
                        - $pi$: A different action rule apart from the two mentioned above. $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$."""
                )
            elif first_run_proposer == "coin_flip":
                task_procedure_description = textwrap.dedent(
                    """\
                    If the sender is the proposer:
                        1. The sender determines a signaling scheme $varphi$ and commits it to the receiver. $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
                        2. The receiver decides an action rule: 
                            - $pi_0$: The receiver ignores the sender's signals and chooses the best response to the prior belief at each time in the sample phase.
                            - $pi_1$: The receiver calculates its posterior belief (using prior belief, the sender's signaling scheme, and every sent signal in the sample phase), and chooses the best response to the posterior belief.
                            - $pi$: A different action rule apart from the two mentioned above. $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$.
                    If the receiver is the proposer:
                        1. The receiver announces a signaling scheme $varphi_1$, claiming that it will follow $pi_1$ if the sender commits to a signaling scheme $varphi$ that yields an expected reward for the receiver at least as high as that induced by $varphi_1$; otherwise, the receiver will follow $pi_0$.
                        2. The sender determines a signaling scheme $varphi$."""
                )
        elif duration == "long_term":
            if long_term_type == "fixed_role":  # first_run_proposer must be system_assigned
                task_procedure_description = textwrap.dedent(
                    """\
                    The following process continues until one of two conditions is met: either the receiver takes $pi_1$ or the game ends due to a timeout:
                        1. The sender determines a signaling scheme $varphi$ and commits it to the receiver. $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
                        2. The receiver decides an action rule: 
                            - $pi_0$: The receiver ignores the sender's signals and chooses the best response to the prior belief at each time in the sample phase.
                            - $pi_1$: The receiver calculates its posterior belief (using prior belief, the sender's signaling scheme, and every sent signal in the sample phase), and chooses the best response to the posterior belief.
                            - $pi$: A different action rule apart from the two mentioned above. $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$."""
                )
            elif long_term_type == "alternating_offer":  # first_run_proposer must be coin_flip
                task_procedure_description = textwrap.dedent(
                    """\
                    - If the sender is the proposer (and the receiver is the responder):
                        - The sender determines a signaling scheme $varphi$ and commits it to the receiver. $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
                        - The receiver decides an action rule: 
                            - $pi_0$: The receiver ignores the sender's signals and chooses the best response to the prior belief at each time in the sample phase.
                            - $pi_1$: The receiver calculates its posterior belief (using prior belief, the sender's signaling scheme, and every sent signal in the sample phase), and chooses the best response to the posterior belief.
                            - $pi$: A different action rule apart from the two mentioned above. $pi: Sigma to Delta(A)$, where $Sigma$ is the sender's signal space, $A$ is the receiver's action space, and $Delta(A) is the set of all random variables on $A$.
                    - If the receiver is the proposer (and the sender is the responder):
                            - The receiver announces a signaling scheme $varphi_1$, claiming that it will follow $pi_1$ if the sender commits to a signaling scheme $varphi$ that yields an expected reward for the receiver at least as high as that induced by $varphi_1$; otherwise, the receiver will follow $pi_0$.
                            - The sender determines a signaling scheme $varphi$

                    The procedure is as follows:
                    1. Who to be the proposer (in the first run) is determined by a coin flip.
                    2. The following process continues until one of three conditions is met: either a consensus is reached (the receiver decides $pi_1$ as a responder or the sender decides a a signaling scheme $varphi$ that yields an expected reward for the receiver at least as high as that induced by $varphi_1$) or the game ends due to a timeout:
                        3. The proposer decides its policy
                            - If the sender is the proposer: The sender determines a signaling scheme $varphi$ and commits it to the receiver. $varphi: S to Delta(Sigma)$, where $S$ is the environmental state space, $Sigma$ is the sender's signal space, and $Delta(Sigma) is the set of all random variables on $Sigma$.
                            - If the receiver is the proposer: The receiver announces a signaling scheme $varphi_1$, claiming that it will follow $pi_1$ if the sender commits to a signaling scheme $varphi$ that yields an expected reward for the receiver at least as high as that induced by $varphi_1$; otherwise, the receiver will follow $pi_0$.
                        4. The responder decides its policy
                            - If the receiver is the responder: The receiver decides an action rule
                            - If the sender is the responder: The sender determines a signaling scheme $varphi$
                        5. If they did not reach a consensus, the two agents switch roles: the current responder becomes the proposer, and the current proposer becomes the responder."""
                )

        task_procedure_description += textwrap.dedent(
            """\
            
            Next, a simulation takes place where the players do not make any new decisions. The environment samples $n$ states, and the players act according to their predefined policies, receiving their corresponding rewards.
            1. The following process continues until $n$ states are sampled:
                2. The environment samples a state $s$ according to the prior state distribution $mu_0$.
                3. The sender signals $sigma$ based on the committed signaling scheme $varphi$.
                4. The receiver selects an action $a$ according to the decided action rule $pi$.
                5. Each agent receives a reward based on the sampled state $s$ and the action $a$ taken by the receiver."""
        )

    # ========================================
    # note that

    if duration == "one_shot":
        if may_meet_again_context == "never_meet_again":
            may_meet_again_type_or_long_term_termination_t_description = (
                "You two will only play this game once. You will not have any interaction with it afterwards."
            )
        elif may_meet_again_context == "may_meet_again_fixed_role":
            may_meet_again_type_or_long_term_termination_t_description = "You two will play this game once. But note that you two might play this game again in the future, with the same role assignment (proposer and responder)."
        elif may_meet_again_context == "may_meet_again_alternating":
            may_meet_again_type_or_long_term_termination_t_description = "You two will play this game once. But note that you two might play this game again in the future, and your roles (proposer and responder) may switch. While the other private properties remains the same (e.g. your agent indices)."

    else:
        if long_term_type == "fixed_role":
            may_meet_again_type_or_long_term_termination_t_description = (
                "The loop process terminates when the timestep equals 5. The initial timestep is 0 and increments by 1 each iteration."
            )
        elif long_term_type == "alternating_offer":
            may_meet_again_type_or_long_term_termination_t_description = "The loop process has a 0.1 probability of stopping each time it is executed. The initial timstep is 0, and it increases by 1 each time it is executed. If the timestep equals 10, it will stop directly."

    may_meet_again_type_or_long_term_termination_t_description = "Note that:\n" + may_meet_again_type_or_long_term_termination_t_description

    # ========================================
    # format

    if task == "bargaining":
        proposer_decision_format = textwrap.dedent(
            """\
            {{
                "Analysis": "(Your Summarized Analysis)", 
                "Decision": x,
            }}
            where $x$ is your decision. It specifies the amount that you decide to leave for yourself. It should be in the range as specified before."""
        )
        responder_decision_format = textwrap.dedent(
            """\
            {{
                "Analysis": "(Your Summarized Analysis)", 
                "Decision": y,
            }}
            where $y$ is your decision and it is either 0 or 1. It should be an integer."""
        )
    elif task == "signaling":
        proposer_decision_format = textwrap.dedent(
            """\
            If you are the sender:
            {{
                "Analysis": "(Your Summarized Analysis)", 
                "Decision": [x1, x2],
            }}
            where:
            - x1 represents $varphi(sigma=1 | s=0)$: the probability of sending signal 1 when the state is 0.
            - x2 represents $varphi(sigma=1 | s=1)$: the probability of sending signal 1 when the state is 1.
            - If you are the sender, this decision specifies your signaling scheme.
            - If you are the receiver, this decision specifies the signaling scheme $varphi_1$ you expect the sender to take, claiming that you will follow $pi_1$ if the sender commits to a signaling scheme $varphi$ that yields an expected reward for the receiver at least as high as that induced by $varphi_1$; otherwise, the receiver will follow $pi_0$."""
        )

        responder_decision_format = textwrap.dedent(
            """\
            {{
                "Analysis": "(Your Summarized Analysis)", 
                "Decision": [y1, y2],
            }}
            If you are the receiver:
                - y1 represents $pi(a=1 | sigma=0)$: the probability of taking action 1 when the signal is 0.
                - y2 represents $pi(a=1 | sigma=1)$: the probability of taking action 1 when the signal is 1.
                - This decision specifies your action rule.
            If you are the sender:
                - x1 represents $varphi(sigma=1 | s=0)$: the probability of sending signal 1 when the state is 0.
                - x2 represents $varphi(sigma=1 | s=1)$: the probability of sending signal 1 when the state is 1.
                - This decision specifies your signaling scheme. You can make it the same as the receiver proposed or any othor signaling scheme."""
        )

    # ========================================
    # init_system_prompt

    formatted_template = (
        init_system_prompt_framework.replace("{role_type1}", "${role_type1}")
        .replace("{role_type2}", "${role_type2}")
        .replace("{task_scenario_description}", "${task_scenario_description}")
        .replace("{reward_function_description}", "${reward_function_description}")
        .replace("{task_procedure_description}", "${task_procedure_description}")
        .replace("{may_meet_again_type_or_long_term_termination_t_description}", "${may_meet_again_type_or_long_term_termination_t_description}")
        .replace("{proposer_decision_format}", "${proposer_decision_format}")
        .replace("{responder_decision_format}", "${responder_decision_format}")
        .replace("$x", "$$x")
    )

    partial_template = string.Template(formatted_template)

    # print(partial_template.template)

    init_system_prompt = partial_template.safe_substitute(
        role_type1=role_type1,
        role_type2=role_type2,
        task_scenario_description=task_scenario_description,
        reward_function_description=reward_function_description,
        task_procedure_description=task_procedure_description,
        may_meet_again_type_or_long_term_termination_t_description=may_meet_again_type_or_long_term_termination_t_description,
        proposer_decision_format=proposer_decision_format,
        responder_decision_format=responder_decision_format,
    )

    # ========================================

    return init_system_prompt


import csv

input_file = "all_possible_tasks.csv"
output_file = "all_possible_init_system_prompt.txt"

with open(input_file, newline="", encoding="utf-8") as csvfile, open(output_file, "w", encoding="utf-8") as outfile:
    reader = csv.reader(csvfile)
    headers = next(reader)

    for row in reader:
        task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type = row

        result = initialize_system_prompt(task, duration, scenario, value_setting, first_run_proposer, may_meet_again_context, long_term_type)

        outfile.write(f"# ========================================\nSettings:\n{dict(zip(headers, row))}\n\nOutput:\n{result}\n\n")

print(f"Results have been saved to {output_file}")
