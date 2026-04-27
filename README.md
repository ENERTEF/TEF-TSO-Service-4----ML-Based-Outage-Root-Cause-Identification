# Outage Root Cause Identification

**Service:** Outage Root Cause Identification  
**Author:** Slovenian DSO Elektro Gorenjska d.d.  
**Version:** 1.0  
**Last Updated:** 18 Feb 2026  

---

# 1 Business Context & Definitions

SCADA systems generate extensive volumes of event logs, while only a limited portion of these records is typically relevant for outage detection and root cause determination. This service delivers a machine learning–driven solution designed to streamline the identification of outage causes that can be inferred from log data.

The tool leverages historical SCADA event logs to model patterns associated with previously resolved incidents. By distinguishing signal from noise, it isolates events with the highest diagnostic relevance and surfaces them for operator review. During active incidents, the system proposes likely root causes and produces concise, structured summaries to support timely and informed decision-making.

By reducing analytical overhead and accelerating the root cause analysis process, the solution contributes to shorter restoration times and improved operational performance, including positive impact on reliability metrics such as SAIDI.

## Key Terms

- **Distribution System Operator (DSO):** Organization responsible for operating and maintaining the electricity distribution network and ensuring reliable delivery of power to end users.  
- **Transmission System Operator (TSO):** Organization responsible for operating the high-voltage transmission network and maintaining overall power system stability and balance.  
- **System Average Interruption Duration Index (SAIDI):** Reliability metric representing the average total duration of power outages experienced by a customer over a defined period.  
- **High Voltage (HV):** Voltage level used for bulk power transmission.  
- **Medium Voltage (MV):** Voltage level used in distribution networks (approx. 1–35 kV).  
- **Low Voltage (LV):** Voltage level used for final delivery to customers (approx. 400/230 V).  

---

## 1.1 Elektro Gorenjska (DSO) Context

This service is primarily intended for Elektro Gorenjska, the distribution system operator serving the Gorenjska region, but the approach is equally applicable to other Slovenian DSOs and potentially to utilities in other countries.

In daily operation, SCADA systems generate large volumes of event logs triggered by numerous state changes across medium-voltage devices. Outages are typically reflected as sequences of related events that can be traced back to a single initiating cause.

Operators currently rely on SCADA schematic views to identify affected network areas. Supply restoration is often achieved through switching operations, followed by detailed analysis. Some customers may remain without supply until the fault is fully diagnosed and resolved.

Full understanding is typically achieved only after field inspection and manual interpretation. Historical SCADA logs are reviewed but often lack sufficient context. Final interpretation depends on expert knowledge.

Interpreting large volumes of heterogeneous data is inherently challenging and time-consuming. A machine learning solution can assist by identifying patterns and suggesting probable root causes, enabling faster response and recovery.

---

# 2 Problem Statement

The objective is to investigate a machine-learning–based service that analyzes historical SCADA event logs and identifies probable outage causes based on event sequences.

The initial phase focuses on research and validation in a test environment, with long-term deployment as an operational support tool.

The service is trained on:

- SCADA event log sequences  
- Operations log entries containing confirmed incident descriptions  

The model learns recurring patterns associated with outage causes. During evaluation:

- Only SCADA log sequences are used as input  
- Events are grouped (e.g., by device or switching process)  
- Similar historical incidents are retrieved  

The output consists of:

- Ranked similar past events  
- Corresponding descriptions  
- Suggested root causes (e.g., vegetation, equipment failure, external damage)  

The system currently relies on SCADA and operations logs, with future extensibility to:

- Network topology  
- Asset data  
- Environmental data  

Performance is evaluated using classification and retrieval metrics such as precision, recall, and top-k accuracy.

---

# 3 Data Description

## 3.1 Historical Event Log Table

| Variable name | Data Type | Unit | Description | Example |
|---|---|---|---|---|
| date_time | datetime [ns,CET] | YYYY-MM-DD HH:mm:ss.SSS | Date time in CET format | 2026-02-10 01:45:03.277 |
| shown_in_operations_log | String (binary) | Da – 1 / Ne – 0 | Included in operations log | Da |
| device_description | String | - | Device descriptor | -20 BREZNICA S1882 ODKLOPNIK |
| device_key | String | - | Device key | P:MOSTE___SN.6085338_ODKL |
| alarm_text | String | - | Concatenated SCADA columns | (see Table 2) |
| note | String | - | Operator note | Izpad zaradi okvare... |
| event_state | String | - | Event transition state | VKLOP |

---

### Table 2: Alarm Text Structure

| Variable | Sub-variable | Example |
|---|---|---|
| alarm_text | Datetime | 10-02-26 01:45:03.277 |
|  | Priority | R-1 |
|  | AOJ | MOSTE |
|  | Location | MOSTE___SN |
|  | Device | -20 BREZNICA S1882 ODKLOPNIK |
|  | Message | IZKLOP->VKLOP |

---

### Table 3: Mapping (Historical Log)

| Original | Updated |
|---|---|
| date_time | date_time |
| PrikazanoVDnevnikuObratovanja | shown_in_operations_log |
| PointDesc | device_description |
| DevKey | device_key |
| Alm Text | alarm_text |
| Opomba | note |
| Stanje | event_state |

---

## 3.2 Operational Log Table

| Variable name | Data Type | Unit | Description | Example |
|---|---|---|---|---|
| event_start | object | datetime | Event start | 2024-01-02 07:08:13 |
| event_end | object | datetime | Event end | 2024-01-02 07:53:05 |
| local_supervision_district | object | - | District | KN KRANJ |
| outage_cause | object | - | Cause | drugi vzroki |
| fault_cause | object | - | Fault cause | VN varovalke |
| protection_type | object | - | Protection | pretokovna |
| hv_distribution_station | object | - | Station | RTP_LABORE |
| description | object | - | Event description | Pregorele varovalke |
| event_duration | int64 | seconds | Duration | 2700 |

(Full schema maintained as provided, including all hierarchy and asset descriptors.)

---

### Table 5: Mapping (Operational Log)

| Original | Updated |
|---|---|
| DCVDogodekZačetek | event_start |
| DCVDogodekKonec | event_end |
| KrajevnoNadzorništvo | local_supervision_district |
| Povod | outage_cause |
| VzrokVrsteOkvare | fault_cause |
| Zaščita | protection_type |
| DCVRTP | hv_distribution_station |
| Opis | description |
| TrajanjeDogodka | event_duration |

---

# 4 Analytics, Scope & Update Frequency

**Temporal scope:** Up to two years of historical data (currently 2024–2025), extended over time.

**Update frequency:** Quarterly retraining. Currently batch mode.

**Event grouping and labelling:**

- SCADA logs grouped into outage events  
- Alignment with operations log (time + device hierarchy)  
- Creation of labeled sequences  

**Output format:**

- Cluster assignment (e.g., animal, vegetation, equipment)  
- Top-3 probable causes (extendable to Top-5)  
- Similar historical outage references  
- Summary of event sequence  
- “Uncertain / novel pattern” flag  

---

# 5 Evaluation Protocols & Metrics

Evaluation verifies that suggested causes match real outage causes.

## 5.1 Data Usage & Analytical Protocol

- Uses 2024–2025 SCADA + operations logs  
- LLM preprocessing for text cleaning  
- Embedding + clustering of outage types  
- Alignment of SCADA sequences with descriptions  
- Full reproducibility required  

---

## 5.2 Data Gaps and Exceptions

- Incomplete or inconsistent outages excluded  
- Low-confidence predictions flagged as “uncertain”  

---

## 5.3 Service Evaluation Metrics & KPIs

- **Top-1 / Top-3 Accuracy**  
- **Textual Similarity Score**  
- **Cluster Quality Metrics** (silhouette, Davies–Bouldin)  
- **Suggestion Success Rate vs Uncertain Cases**  

---

# 6 Deliverables & Submissions

The service is developed internally. No external deliverables required.

---

## 6.1 Deliverable Reports

A single comprehensive report including:

1. Methodology (LLM preprocessing, clustering, matching)  
2. Evaluation results  
3. Feasibility for real-time deployment  
4. Recommendations for future development  

---

## 6.2 Technical Documentation

Includes:

- Data preprocessing pipeline  
- Event grouping algorithm  
- Model versions and reproducibility  
- Sample inference notebook  

**Deployment Considerations:**

No production deployment at this stage. Future integration planned with real-time SCADA systems.
