import json, time, os
from pathlib import Path
from typing import Literal,Any

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from openrouter import OpenRouter
from pydantic import BaseModel, Field, SecretStr
import time
import rich
from rich.markdown import Markdown
import matplotlib.pyplot as plt

from openai.types.chat import ChatCompletionMessageParam

load_dotenv(override=True)

openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
if openrouter_api_key is None:
    raise ValueError("OPENROUTER_API_KEY is missing")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


openrouter = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key, timeout=120, max_retries=5)

DATA = Path("data")

def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))

def get_sample_data(tiny=False, held_out=False):
    """Returns a list of claims, and the handler's decisions on them keyed by claim id
    
    held_out=True gives the twenty-five held-out claims in place of the queue.
        tiny=True keeps the five claims marked for the tiny version of either set.
    """

    claims = load("evals/held_out_claims.json" if held_out else "claims.json")
    decisions = load("evals/held_out_decisions.json" if held_out else "evals/decisions.json")

    if tiny:
        chosen = {decision["claim_id"] for decision in decisions if decision["tiny"]}
        claims = [claim for claim in claims if claim["claim_id"] in chosen]
        decisions = [decision for decision in decisions if decision["tiny"]]
    labels = {decision["claim_id"]: decision for decision in decisions}
    return claims, labels

TINY = True

claims, labels = get_sample_data(tiny=TINY)
policies = {policy["policy_id"]: policy for policy in load("policies.json")}
policy_wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")

# rich.print(Markdown(claims[0]))
# print(json.dumps(labels[claims[0]["claim_id"]], indent=2))

rows=[]

for claim in claims:
    label = labels[claim["claim_id"]]
    rows.append({
        "claim_id": claim["claim_id"],
        "summary": claim["summary"],
        "claim_type": claim["claim_type"],
        "amount_claimed": claim["amount_claimed"],
        "difficulty": label["difficulty"],
        "decision": label["decision"],
        "payout": label["payout"]
    })

data = pd.DataFrame(rows)
# print(data)

# claim_counts= pd.crosstab(
#     data["claim_type"],
#     data["decision"]
# )

# claim_counts.plot(
#     kind="bar",
#     stacked=True,
#     figsize=(10,6)
# )

# plt.title("Claims by Type")
# plt.xlabel("Claim type")
# plt.ylabel("number of claims")

# plt.xticks
# plt.tight_layout()
# plt.show()

# fig = px.histogram(
#     data,
#     x="claim_type",
#     color="decision",
#     title="Claims by type"
# )

# fig.show()

INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
Decide the claim you are given, using the policy wording below and the customer's policy record.

Choose one decision:
- approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
- decline: the wording does not cover the claim.
- refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.
"""

SYSTEM_PROMPT=f"""
{INSTRUCTIONS}
{policy_wording}
"""

AGENT_SYSTEM_PROMPT = f"""{INSTRUCTIONS}
Use your tools to look up the customer's policy record and to record your decision.

{policy_wording}
"""

class ClaimDecision(BaseModel):
    """A Claims handler's decision on one claim"""

    rational: str = Field(description="2-3 sentences to explain the decision along with the relevant section of the policy")
    decision: Literal["approve","decline","refer"]
    payout: float = Field(description="The amount to pay in US dollars, or 0 for a decline or a referral")

def make_user_prompt(claim):
    """The customer's policy record and the claim, as JSON."""
    policy = policies[claim["policy_id"]]

    return f"""
    #Policy Record
    {json.dumps(policy, indent=2)}

    #Claim
    {json.dumps(claim, indent=2)}
    """



def make_message(claim: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    """Build the messages for one claim: the task and the wording, then the policy record and the claim."""
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(claim)},
    ]

def decide(claim, model, effort):
    """Ask one model to decide one claim."""
    start = time.time()

    response = openrouter.chat.completions.parse(
        model=model,
        messages=make_message(claim),
        response_format=ClaimDecision,
        reasoning_effort=effort,
        extra_body={
            "provider": {
                "require_parameters":True
            }
        }
    )
    answer = response.choices[0].message.parsed

    if answer is None:
        raise ValueError("The model did not return a valid ClaimDecision")


    usage = response.usage
    cost = (
        getattr(usage, "cost", None)
        if usage is not None
        else None
    )
    return {
        "model" : model,
        "effort": effort,
        "technique": "LLM",
        "decision": answer.decision,
        "payout": answer.payout,
        "rational": answer.rational,
        "cost": cost,
        "seconds": round(time.time() - start , 1)
    }


llm = ChatOpenAI(
    model="gpt-6-luna",
    api_key=SecretStr(openrouter_api_key),
    base_url=OPENROUTER_BASE_URL
)



def make_langchain_message(claim):
    """Build the langchain messages for one claim: the task and the wording, then the policy record and the claim."""
    return[
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(make_user_prompt(claim))
    ]

# messages = make_langchain_message(claims[0])

# response = llm.invoke(messages)
# print(response.content)

recorded ={}

@tool
def get_policy(policy_id: str) -> str:
    """Look up a customer's policy record: the plan, the add-ons, the purchase date, the trip and the travelers."""
    policy =policies.get(policy_id)
    return json.dumps(policy, indent=2) if policy else f"There is no policy {policy_id}"

@tool
def record_decision(
        claim_id: str, 
        decision: Literal["approve", "decline", "refer"],
        payout: float,
        rationale: str
    ) -> str :
    """Record the decision on a claim in the claims system.

    payout is the amount to pay in US dollars, or 0 for a decline or a referral.
    rationale is 2-3 sentences to explain the decision along with the relevant section of the policy.
    """
    recorded[claim_id]= {
        "decision":decision,
        "payout": payout,
        "rationale": rationale
    }    

    return f"Recorded {decision} on {claim_id}"


luna = ChatOpenAI(model="gpt-6-luna", api_key=SecretStr(openrouter_api_key), base_url=OPENROUTER_BASE_URL)

AGENT_SYSTEM_PROMPT = f"""{INSTRUCTIONS}
Use your tools to look up the customer's policy record and to record your decision.

{policy_wording}
"""

def make_claim_prompt(claim):
    """The claim alone, as JSON. The agent looks up the policy record for itself."""
    return f"""# Claim

{json.dumps(claim, indent=2)}
"""

agent = create_agent(model=luna, tools=[get_policy, record_decision], system_prompt=AGENT_SYSTEM_PROMPT)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": make_claim_prompt(claims[0])}
        ]
    }
)

for message in result["messages"][1:]:
    message.pretty_print()