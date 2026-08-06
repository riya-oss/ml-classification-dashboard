# Dataset Notes

## Overview

| | |
|---|---|
| **Dataset Name** | Telco Customer Churn — IBM Dataset |
| **Source** | [Kaggle: Telco customer churn: IBM dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=download) |
| **Records** | 7,043 |
| **Features** | 33 |
| **Target Variable** | `Churn Label` (Yes / No) — also available as `Churn Value` (1 / 0) |
| **Missing Values** | `Churn Reason` only (5,174 nulls — blank for non-churned customers) |

---

## Data Dictionary

### Customer Identity & Location

| Column | Type | Description |
|---|---|---|
| `CustomerID` | string | Unique customer identifier |
| `Count` | int | Always 1 — used for aggregation |
| `Country` | string | Country of residence |
| `State` | string | US state |
| `City` | string | City of residence |
| `Zip Code` | int | Postal code |
| `Lat Long` | string | Combined lat/long string |
| `Latitude` | float | Geographic latitude |
| `Longitude` | float | Geographic longitude |

### Demographics

| Column | Type | Description |
|---|---|---|
| `Gender` | string | Male / Female |
| `Senior Citizen` | string | Whether customer is 65+ (Yes / No) |
| `Partner` | string | Has a partner (Yes / No) |
| `Dependents` | string | Has dependents (Yes / No) |

### Services Subscribed

| Column | Type | Description |
|---|---|---|
| `Tenure Months` | int | Months with the company |
| `Phone Service` | string | Has phone service (Yes / No) |
| `Multiple Lines` | string | Multiple phone lines (Yes / No / No phone service) |
| `Internet Service` | string | DSL / Fiber optic / No |
| `Online Security` | string | Online security add-on (Yes / No / No internet) |
| `Online Backup` | string | Online backup add-on (Yes / No / No internet) |
| `Device Protection` | string | Device protection add-on (Yes / No / No internet) |
| `Tech Support` | string | Tech support add-on (Yes / No / No internet) |
| `Streaming TV` | string | TV streaming (Yes / No / No internet) |
| `Streaming Movies` | string | Movie streaming (Yes / No / No internet) |

### Account & Billing

| Column | Type | Description |
|---|---|---|
| `Contract` | string | Month-to-month / One year / Two year |
| `Paperless Billing` | string | Paperless billing enabled (Yes / No) |
| `Payment Method` | string | Electronic check / Mailed check / Bank transfer / Credit card |
| `Monthly Charges` | float | Current monthly bill (USD) |
| `Total Charges` | object | Total billed to date (USD) — contains blanks for new customers |

### Target & Churn Metadata

| Column | Type | Description |
|---|---|---|
| `Churn Label` | string | **Target** — whether the customer churned (Yes / No) |
| `Churn Value` | int | Binary encoding of Churn Label (1 = churned, 0 = retained) |
| `Churn Score` | int | IBM-assigned churn propensity score (0–100) |
| `CLTV` | int | Customer Lifetime Value (predicted, USD) |
| `Churn Reason` | string | Customer-stated reason for leaving — null if not churned |
