import json
from pathlib import Path
import pandas as pd

DATA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "claims.json"
)

with DATA.open(encoding="utf-8") as file:
    claims = json.load(file)

# print("File: ", DATA)
# print("Type of Clain: ", type(claims))
# print("Number of Claims: ", len(claims))

first_claim= claims[0]

# print("type of 1 claim: ", type(first_claim))

# print("files of 1 clain: ")

# for field in first_claim.keys():
#     print("-", field)

# print("field type: ")

# for field, value in first_claim.items():
#     print(f"{field:20} - {type(value)}")

# items = first_claim["items"]

# print(f"type of item: {type(items).__name__}")
# print(f"Number of items {len(items)}")

# first_item=items[0]

# print(f"first item {type(first_item).__name__}")
# print(f"first item fields")

# for field, value in first_item.items():
#     print(f"{field:15} - {value!r}")

claims_df = pd.DataFrame(claims)

# print("DF information")
# print("type: ", type(claims_df).__name__)
# print("Shape: ", claims_df.shape)

# print("first 3 claims: ")
# print(
#     claims_df[
#         [
#         "claim_id",
#         "claim_type",
#         "amount_claimed",
#         "status"
#         ]
#     ]
#     .head(3)
#     .to_string(index=False)
# )


# print("Claim Info")
# claims_df.info()

claims_df['submitted_on'] = pd.to_datetime(
    claims_df['submitted_on']
)

claims_df['incident_date'] = pd.to_datetime(
    claims_df['incident_date']
)

# print(
#     claims_df[
#         ["submitted_on", "incident_date"]
#     ].head()
# )

# print("Updated datatypes: ")
# print(
#     claims_df[
#         ["submitted_on", "incident_date"]
#     ].dtypes
# )

# print(claims_df["amount_claimed"].sum())

import matplotlib.pyplot as plt

amount_by_type = (
    claims_df
    .groupby("claim_type", as_index=False)[["amount_claimed"]]
    .sum()
    .sort_values(
        by="amount_claimed",
        ascending=False
    )
)
# plt.figure(figsize=(10,6))

# plt.bar(
#     amount_by_type["claim_type"],
#     amount_by_type["amount_claimed"]
# )

# plt.title("Total Amount by claim type")
# plt.xlabel("Claim Type")
# plt.ylabel("Amount Claimed")

# plt.xticks(rotation=45, ha="right")

# plt.tight_layout()

# plt.show()

status_count = claims_df["status"].value_counts()

plt.figure(figsize=(7,4))

plt.hist(
    claims_df["amount_claimed"],
    bins=10,
    edgecolor="black"
)

plt.show()