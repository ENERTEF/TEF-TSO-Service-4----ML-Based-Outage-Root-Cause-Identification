# Outage Root Cause Identification

**Service:** Outage Root Cause Identification  
**Document Type:** Technical Manual & Service Specification  
**Author:** Slovenian DSO Elektro Gorenjska d.d.  
**Version:** 1.0  
**Last Updated:** 18 Feb 2026  

---

# 1. Business Context & Definitions

SCADA systems generate extensive volumes of event logs, while only a limited portion of these records is typically relevant for outage detection and root cause determination. This service delivers a machine learning–driven solution designed to streamline the identification of outage causes that can be inferred from log data.

The tool leverages historical SCADA event logs to model patterns associated with previously resolved incidents. By distinguishing signal from noise, it isolates events with the highest diagnostic relevance and surfaces them for operator review. During active incidents, the system proposes likely root causes and produces concise, structured summaries to support timely and informed decision-making.

By reducing analytical overhead and accelerating the root cause analysis process, the solution contributes to shorter restoration times and improved operational performance, including positive impact on reliability metrics such as SAIDI.

## Key Terms

| Term | Definition |
|---|---|
| Distribution System Operator (DSO) | Organization responsible for operating and maintaining the electricity distribution network and ensuring reliable delivery of power to end users |
| Transmission System Operator (TSO) | Organization responsible for operating the high-voltage transmission network and maintaining overall power system stability and balance |
| System Average Interruption Duration Index (SAIDI) | Reliability metric representing the average total duration of power outages experienced by a customer over a defined period, usually expressed in minutes per year |
| High Voltage (HV) | Voltage level used for bulk power transmission over long distances, typically tens to hundreds of kilovolts |
| Medium Voltage (MV) | Voltage level used in distribution networks to supply substations and large consumers, typically in the range of about 1–35 kV |
| Low Voltage (LV) | Voltage level used for final delivery of electricity to residential and small commercial customers, typically around 400/230 V in European systems |

---

## 1.1 Elektro Gorenjska DSO Context

This service is primarily intended for **Elektro Gorenjska**, the distribution system operator (DSO) serving the Gorenjska region, but the approach is equally applicable to other Slovenian DSOs, including those serving Ljubljana, Koper (Primorska region), Maribor, and Celje, and potentially to utilities in other countries.

In daily operation, SCADA systems generate large volumes of event logs triggered by numerous state changes across medium-voltage (MV) SCADA-compatible devices. Outages are typically reflected as sequences of related events that can, in principle, be traced back to a single initiating cause. In current practice, operators rely on the SCADA schematic view to identify the network area in which a fault has been detected. Where network topology allows, supply is restored by switching from the opposite side of a ring connection, after which the incident is analyzed in detail. Despite these measures, some customers may remain without supply until the underlying fault is fully diagnosed and resolved.

A complete understanding of the event is usually obtained only after a field technician inspects the affected location and provides a detailed report. Historical SCADA event logs are also reviewed, although these records often provide only limited contextual information. Final interpretation is therefore performed by experienced operators, who supplement the operational event log with a technical description of the incident.

Interpreting large volumes of heterogeneous data from multiple sources is inherently challenging and time-consuming. A machine learning–based solution could assist by identifying patterns in historical data and presenting the most probable root causes of outages. Such support has the potential to accelerate situational understanding, enabling faster response and recovery.

---

# 2. Problem Statement

The objective is to investigate the feasibility of a machine-learning–based service that analyzes historical SCADA event logs and identifies the most probable causes of power outages based on observed sequences of events. The initial phase focuses on research and validation in a test environment; if the approach proves reliable, the long-term goal is deployment as an operational support tool to improve the efficiency of operators in the control center.

The service will be trained on historical data consisting of SCADA event log sequences and corresponding entries from the operations log, which contain confirmed descriptions of past incidents. The model is expected to learn recurring patterns of events associated with specific outage causes. During evaluation, the system will receive only SCADA log sequences and will group related events, such as those associated with the same device or switching process, before identifying historical incidents with similar patterns. The primary output will be a ranked set of the most similar past events together with their recorded descriptions, providing operators with plausible explanations, for example vegetation contact, equipment failure, or external damage.

At this stage, the system will rely exclusively on SCADA event logs and operations log records; however, the design should allow for the possible inclusion of additional data sources in the future, such as network topology, asset information, or environmental data. Training may use complete historical records, including post-event descriptions, but inference will be restricted to information available in the event logs themselves.

The prototype will be developed for internal evaluation and will not initially require a production interface. Performance will be assessed by comparing the predicted or retrieved incident descriptions with the documented root causes of historical outages. Standard information-retrieval and classification metrics, such as precision, recall, or top-k accuracy, can be applied, with the possibility of semi-automated similarity assessment to support large-scale evaluation.

---

# 3. Data Description

## 3.1 Data Table of Historical Event Log

Table 1 summarises the available features of the dataset for the historical event log.

### Table 1: Historical Event Log Table

| Variable name | Data Type | Unit | Description | Example |
|---|---|---|---|---|
| `date_time` | datetime [ns,CET] | YYYY-MM-DD HH:mm:ss.SSS | Date time in CET format | 2026-02-10 01:45:03.277 |
| `shown_in_operations_log` | String (binary choice) | Da – 1 / Ne – 0 | Is the log included in the operations log SQL table | Da |
| `device_description` | String | - | Device name and informative descriptor | -20 BREZNICA S1882 ODKLOPNIK |
| `device_key` | String | - | Device key | P:MOSTE___SN.6085338_ODKL |
| `alarm_text` | String | - | Concatenated columns data, explained in Table 2 | 10-02-26 01:45:03.277 R-1 MOSTE MOSTE___SN -20 BREZNICA S1882 ODKLOPNIK IZKLOP->VKLOP Na objektu |
| `note` | String | - | Operators note, only when row is included in operations log | Izpad zaradi okvare na DV med TP Mlaka in stikalom S0134 (kuna na sitkalu) |
| `event_state` | String | - | To what state the event transitioned | VKLOP |

Since Table 1 features a variable with concatenated columns data, Table 2 offers additional explanation of the `alarm_text` variable. An example of how historical event logs are originally stored in the SCADA table is shown in Figure 1, while Table 3 shows how original column names are mapped to the updated English named columns.

### Table 2: Sub-Variables Explanation from Historical Event Log Table, `alarm_text` Variable

| Variable name | Sub-variable | Example text |
|---|---|---|
| `alarm_text` | Datetime | 10-02-26 01:45:03.277 |
| `alarm_text` | Priority | R-1 |
| `alarm_text` | AOJ | MOSTE |
| `alarm_text` | Location | MOSTE___SN |
| `alarm_text` | Device / Description | -20 BREZNICA S1882 ODKLOPNIK |
| `alarm_text` | Message | IZKLOP->VKLOP Na objektu |

### Table 3: Mapping Table Used with Historical Event Log Table

| Original column name | Updated column name |
|---|---|
| `date_time` | `date_time` |
| `PrikazanoVDnevnikuObratovanja` | `shown_in_operations_log` |
| `PointDesc` | `device_description` |
| `DevKey` | `device_key` |
| `Alm Text` | `alarm_text` |
| `Opomba` | `note` |
| `Stanje` | `event_state` |

---

## 3.2 Data Table of Operational Log

Table 4 offers an in-depth operational log, which features hierarchical device structure and can be timestamp coaligned with historical event log table data, while Table 5 lists a mapping table of original database column names and the updated standardised English names.

### Table 4: Operational Log Table

| Variable name | Data Type | Unit | Description | Example |
|---|---|---|---|---|
| `var_en` | `data_type` | `unit` | `description` | `example` |
| `event_start` | object | datetime | Event start datetime | 2024-01-02 07:08:13.000 |
| `event_end` | object | datetime | Event end datetime | 2024-01-02 07:53:05.000 |
| `local_supervision_district` | object | - | Local supervision district name | KN KRANJ |
| `outage_cause` | object | - | Cause/reason for the outage | drugi vzroki |
| `fault_cause` | object | - | Cause of fault | VN varovalke |
| `protection_type` | object | - | Protection mechanism type | pretokovna |
| `hv_distribution_station` | object | - | HV/MV distribution station code | RTP_LABORE |
| `fault_cause_secondary` | object | - | Secondary fault cause | VN varovalke |
| `event_category` | object | - | Event category/group | drugi vzroki |
| `disturbance_cause` | object | - | Disturbance cause description | lastni vplivi |
| `fault_type` | object | - | Type of fault | ostali elementi in naprave |
| `description` | object | - | Event description | Pregorele varovalke, kuna na TR |
| `hv_feeder_code` | object | - | HV/MV feeder code | LAB_20_OREHEK |
| `hv_device_gis_name` | object | - | HV/MV device GIS name | IZV-R01 T0201 FARMA ŽABNICA HLEVI |
| `distributor_name` | object | - | Distribution point name | NNR T0201 FARMA ŽABNICA |
| `hv_bay_voltage_name` | object | - | HV bay voltage level name | E04 TR2 |
| `hv_bay_transformer_name` | object | - | HV bay transformer name | TR RT 40000-110 |
| `transformer_name` | object | - | Transformer name | TR 4HT 100/20/10-0,4 |
| `transformer_station_name` | object | - | Transformer station name | T0201 FARMA ŽABNICA |
| `lv_dist_station_bay_name` | object | - | LV distribution station bay name | J07 LJUBELJ |
| `lv_transformer_bay_name` | object | - | LV transformer station bay name | J21 OREHEK |
| `dist_transformer_station_name` | object | - | Distribution transformer station name | T0983 RP BALOS |
| `gis_name` | object | - | GIS location/device name | S0393 POLJE-FARMA ŽABNICA |
| `gis_device_type` | object | - | GIS device type classification | Ločilno mesto |
| `lv_structure_name` | object | - | LV organizational structure name | TR 4HT 100/20/10-0,4 |
| `feeder_distributor_name` | object | - | Feeder distributor name | IZV-R01 T0201 FARMA ŽABNICA HLEVI |
| `connection_cabinet_name` | object | - | Connection cabinet name | PR_OM ŽABNICA 70 |
| `event_duration` | int64 | seconds | Event duration in seconds | 2700 |

### Table 5: Mapping Table Used with the Operational Log Table Data

| Original col name | Updated col name |
|---|---|
| `var_si` | `var_en` |
| `DCVDogodekZačetek` | `event_start` |
| `DCVDogodekKonec` | `event_end` |
| `KrajevnoNadzorništvo` | `local_supervision_district` |
| `Povod` | `outage_cause` |
| `VzrokVrsteOkvare` | `fault_cause` |
| `Zaščita` | `protection_type` |
| `DCVRTP` | `hv_distribution_station` |
| `VzrokVrsteOkvare1` | `fault_cause_secondary` |
| `Skupina` | `event_category` |
| `Povodmotnja` | `disturbance_cause` |
| `VrstaOkvare` | `fault_type` |
| `Opis` | `description` |
| `DCVIzvod` | `hv_feeder_code` |
| `DCVNapravaSIDGisNaziv` | `hv_device_gis_name` |
| `RazdelilecNaziv` | `distributor_name` |
| `PoljeCelicaVNTRVNNaziv` | `hv_bay_voltage_name` |
| `PoljeCelicaVNTR_TRNaziv` | `hv_bay_transformer_name` |
| `TransformatorNaziv` | `transformer_name` |
| `TransformatorskaPostajaNaziv` | `transformer_station_name` |
| `PoljeCelicaSNRPNaziv` | `lv_dist_station_bay_name` |
| `PoljeCelicaSNRTPNaziv` | `lv_transformer_bay_name` |
| `RazdelilnaTransformatorskaPostajaNaziv` | `dist_transformer_station_name` |
| `GisNaziv` | `gis_name` |
| `GisTipNaprave` | `gis_device_type` |
| `StrukturaSOMNaziv` | `lv_structure_name` |
| `IzvodRazdelilcaNaziv` | `feeder_distributor_name` |
| `PriključnaOmaricaNaziv` | `connection_cabinet_name` |
| `TrajanjeDogodka` | `event_duration` |

---

# 4. Analytics, Scope & Update Frequency

## Temporal Scope

Analytics are computed on up to two years of historical data, currently 2024–2025 SCADA event logs and corresponding operations logs for both outages and planned switch-offs. The historical window will be extended as new years of data become available, ensuring that clustering and pattern discovery always leverage the most recent operational experience.

## Update Frequency

Outage-type clustering models are retrained quarterly, incorporating newly accumulated SCADA logs and operations log records. At this stage, the service operates in batch mode as part of a research and prototyping workflow; real-time deployment is planned for a later phase once stable access to live event streams and robust operational validation are in place.

## Event Grouping and Labelling

For training, the service automatically groups historical SCADA log entries into outage events by aligning event timestamps and device hierarchy with the operations log, which provides confirmed outage intervals and textual descriptions. This grouping effectively “labels” historical event sequences with known outage causes, while additional SCADA events not included in the operations log are associated to the same incident where they fall within the same temporal window and device structure.

## Output Format

For each detected outage event, the service returns a structured set of indicators, including:

- Cluster assignment of the outage, for example animal-related, vegetation-related, human/mechanical damage, equipment fault, or other, together with a short cluster description.
- Top 3 most probable outage causes inferred from the historical event-log patterns, with associated scores or probabilities and the option to later expand to a top-5 list.
- References to the most similar past outages, including operations-log entries and their descriptions, that support the suggested cause.
- A concise summary of the characteristic event sequence, including key alarms, devices, and switching actions that drove the classification, highlighting events not explicitly recorded in the operations log.
- An “uncertain / novel pattern” flag for cases where no existing outage-type cluster matches with sufficient confidence. These events are candidates for operator review and future model refinement, especially relevant when transitioning to new systems such as ADMS.

---

# 5. Evaluation Protocols & Metrics

The purpose of the evaluation is to verify that the service reliably proposes outage causes and similar historical events that are close to the true root causes documented in the operations log. Evaluation focuses on methodological correctness, clustering and sequence modelling, quality of textual suggestions, and their usefulness for operators during incident analysis.

---

## 5.1 Data Usage & Analytical Protocol

- The service uses up to two years of historical data, currently 2024–2025, from SCADA event logs and corresponding operations logs for both outages and planned switch-offs. As new years of data become available, they are incorporated into the training and validation datasets.
- Before modelling, operations log “Event description” fields are cleaned and normalised using a large language model (LLM)-based preprocessing pipeline, which removes low quality entries and harmonises similar descriptions into consistent labels.
- Cleaned descriptions are then embedded and clustered into semantically coherent outage type groups, for example animal contact, vegetation, equipment failure, human/mechanical damage, or other, with cluster centroids or prototype phrases representing typical causes for each group.
- Historical SCADA event sequences are automatically aligned to operations log entries and their cluster labels via time windows and device hierarchy, creating training pairs of “event sequence → canonicalised outage description / cluster”. These pairs form the basis for both clustering refinement and supervised model training.
- All experiments must be reproducible, with documented model versions, LLM preprocessing settings, clustering parameters, training/validation splits, and input datasets.

---

## 5.2 Data Gaps and Exceptions

- Outages with incomplete or inconsistent data, for example missing key parts of the SCADA sequence or unclear or unusable operations-log descriptions after LLM cleaning, are excluded from quantitative accuracy metrics and may only be used for qualitative inspection.
- Events for which the model outputs a very low confidence score or cannot match any existing cluster are flagged as “uncertain / novel pattern” and routed for manual review; these are reported separately rather than counted as standard successes or failures.

---

## 5.3 Service Evaluation Metrics & KPIs

The service is evaluated using quantitative metrics that assess how well predicted causes and text suggestions match the cleaned, canonicalised labels and descriptions derived from the operations log.

| Metric | Description |
|---|---|
| Top-1 and Top-3 Cause Accuracy | Share of outages where the canonicalised ground-truth cause, meaning the cluster label, lies in the model’s top-1 and top-3 suggested causes for the event sequence. This measures how often the system proposes a cause that matches the validated description, for example mapping “Pregorele varovalke, kuna na TR” to an “animal-related fuse blowout” cluster. |
| Textual Similarity Score | Average semantic similarity, for example cosine similarity in embedding space, between the model’s primary suggested description and the cleaned ground-truth event description, computed over the validation set and reviewed on a sample by human experts to verify that “close enough” suggestions are indeed meaningful. |
| Cluster Quality Metrics | Internal clustering metrics, such as silhouette score or Davies–Bouldin index, and external cluster purity with respect to the canonicalised cause labels, to verify that outage-type clusters, including animals, vegetation, human/mechanical, equipment, and other, are coherent and interpretable. |
| Suggestion Success Rate vs. Uncertain Cases | Proportion of evaluated outages that receive a high-confidence, semantically correct suggestion, as judged by the above metrics and spot human review, versus those classified as “uncertain / novel pattern”. This effectively captures “how many cases got good suggestions and how many failed completely.” |

---

# 6. Deliverables & Submissions

The service will be developed and evaluated internally. No external provider deliverables are required.

---

## 6.1 Deliverable Reports

A single comprehensive report documenting:

1. Methodology for operations-log preprocessing, outage-type clustering, and event-sequence pattern matching.
2. Evaluation results on historical 2024–2025 data, including accuracy metrics, cluster quality, and example predictions.
3. Technical feasibility assessment for future real-time deployment, including challenges, requirements, and next steps.
4. Recommendations for further development, including data requirements and integration with ADMS transition.

---

## 6.2 Technical Documentation

## Model & Pipeline Documentation

The model and pipeline documentation will be included in the research report and will cover:

- Data preprocessing pipeline, including LLM cleaning and clustering parameters.
- Event grouping algorithm, based on time-window and device hierarchy matching.
- Model versions, training/validation splits, and reproduction instructions.
- Sample inference notebook demonstrating predictions on new outage sequences.

## Deployment Considerations

No production deployment is planned at this stage. Code and documentation will support future integration with real-time SCADA streams once data access and operational requirements are resolved.
