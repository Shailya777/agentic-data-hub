# Architecture Log

## Date: May 25, 2026
**Phase 0 & Phase 1: Project Initialization & Data Ingestion**
* **Repository Structure:** Established a modular Python package structure separating UI (`src/ui`), orchestration (`src/agents`), and specific operational engines (`src/engines/sql_engine.py`, etc.).
* **Environment:** Configured a local Conda environment (Python 3.10) to manage dependencies cleanly and ensure reproducible local execution.
* **Database Foundation:** Initialized a local MySQL instance (`ecommerce_db`).
* **Data Pipeline:** Developed a Python ingestion script using `pandas` and `SQLAlchemy` (`pymysql` driver) to load the raw Olist E-commerce CSV files into relational SQL tables, processing via 10,000-row chunks for memory safety.

**Phase 1.1: Database Security Setup**
* **Access Control:** Deprecated local `root` access for the application layer. Created a dedicated MySQL user (`agentic_app`) adhering to the Principle of Least Privilege.
* **Privileges:** The application user is restricted to `GRANT ALL PRIVILEGES` strictly within the `ecommerce_db` schema, neutralizing global database threats from potential LLM hallucination or prompt injection in the downstream Text-to-SQL engine.

**Phase 1.2: Database Constraint & Indexing Enforcement**
* **Relational Integrity:** Executed a post-ingestion DDL script (`apply_constraints.sql`) to convert implicit Pandas `TEXT` columns into `VARCHAR`.
* **Performance Optimization:** Explicitly defined Primary Keys and Foreign Keys across `customers`, `orders`, `products`, and `order_items`. This automatically generates B-Tree indexes in MySQL, drastically reducing query latency for the downstream Text-to-SQL agent during complex `JOIN` operations.

**Phase 1.3: Data Ingestion Paradigm Decision (ELT)**
* **Architecture Choice:** Explicitly selected an ELT (Extract, Load, Transform) pattern over traditional ETL.
* **Implementation:** Leveraged Pandas for rapid, schema-agnostic extraction and loading of CSVs into MySQL as raw text. Post-ingestion, executed SQL DDL scripts to transform column types and enforce relational integrity constraints.
* **Justification:** Optimizes for development speed and data resilience. Handling constraints and type-casting within the database engine is a standard practice in modern data architecture, ensuring the ingestion pipeline doesn't break on minor CSV formatting anomalies.

**Phase 1.4: Connection String Hardening**
* **Bug Fix:** Resolved a URL parsing failure caused by special characters (`@`) in the database password. 
* **Implementation:** Refactored the `SQLAlchemy` connection string generation to use the `sqlalchemy.engine.URL.create()` method instead of standard Python f-strings. This ensures all database credentials are automatically and safely URL-encoded before attempting a connection pool initialization.

## Date: May 26, 2026
**Phase 2.2: Multi-Intent Orchestration Upgrade**
* **Architectural Pivot:** Identified a critical limitation in the initial intent router (single-choice classification), which would fail on complex, hybrid business queries.
* **Implementation:** Refactored the Pydantic structured output model from a single string to a `List[str]` and updated the prompt engineering. Replaced standard JSON mode with OpenAI 2.x's native `beta.chat.completions.parse()` method for guaranteed schema adherence. 
* **Impact:** The router can now detect when a query requires parallel execution (e.g., routing to `['SQL_ENGINE', 'RAG_ENGINE']` simultaneously for queries involving both revenue metrics and qualitative sentiment).

## Date: May 27, 2026
**Phase 3.1: Metadata-Driven UI Architecture**
* **Architectural Pivot:** Rejected the standard (and highly insecure) approach of allowing the LLM to write and execute Python `matplotlib` code to render charts.
* **Implementation:** Engineered a Metadata-Driven UI. Upgraded the `SQLResponse` Pydantic model to force the LLM to output a strict JSON schema containing the SQL query alongside explicit charting metadata (chart type, X-axis, Y-axis).

**Phase 4A: Streamlit Human-in-the-Loop Dashboard**
* **Implementation:** Initialized the frontend using Streamlit (`src/ui/app.py`) to serve as the unified presentation and verification layer.
* **Security & Trust:** Implemented a "Human-in-the-Loop" verification step. The UI exposes the AI-generated SQL and charting logic in an expandable code block, allowing data scientists to verify the logic before trusting the rendered visualizations.
* **Dynamic Rendering:** Connected the LLM's metadata JSON directly to Streamlit's native charting functions (`st.bar_chart`, `st.line_chart`, etc.), completely separating the analytical logic (AI) from the visual rendering layer (Frontend).

## Date: May 28, 2026
**Phase 4B: RAG Engine & Cross-Lingual Synthesis**
* **Implementation:** Engineered a local vector pipeline using ChromaDB and OpenAI's `text-embedding-3-small` model to index unstructured customer reviews. 
* **Data Processing:** Implemented targeted data filtering to only embed reviews containing actual text, dropping empty 5-star ratings to optimize vector space and reduce noise.
* **Cross-Lingual Capability:** Leveraged frontier LLM capabilities to perform zero-shot cross-lingual RAG. The system natively reads Portuguese vector context and synthesizes professional, English-only analytical summaries without requiring an explicit translation layer.
* **Prompt Engineering:** Implemented strict system constraints to prevent standard LLM hallucinations (e.g., citing arbitrary chunk IDs like "According to Review 1") to ensure the output reads like a cohesive report from a human Data Analyst.

## Date: May 28, 2026
**Phase 5: The Grand Integration & Query Decomposition**
* **Architectural Challenge:** Identified and resolved "Prompt Leakage," where a single multi-intent query passed in its entirety to all engines caused hallucinations (e.g., the RAG engine apologizing for not finding financial data in text reviews).
* **Implementation:** Upgraded the Intent Router to perform full **Query Decomposition**. Instead of just listing required engines, the orchestrator now splits and rewrites the user's prompt into isolated, engine-specific sub-queries using strict Pydantic schemas. 
* **Execution:** Integrated the orchestrator into the Streamlit UI, allowing parallel execution of the SQL and RAG engines while completely isolating their context windows, resulting in a flawless hybrid dashboard.

## Date: May 31, 2026
**Phase 6: V1.5 Predictive Engine (Machine Learning & MLOps)**
* **Architectural Challenge:** The platform needed to proactively predict logistical failures (Delivery Delays) on highly imbalanced e-commerce data (only an 8% baseline delay rate), requiring a robust, production-safe machine learning pipeline.
* **Feature Engineering & SQL Extraction:** * Engineered a custom volumetric feature (`total_volume_cm3`) to prevent "Orientation Noise" inherent in raw dimensions.
    * Engineered temporal features (`purchase_month`, `purchase_day_of_week`) and calculated a `seller_historical_delay_rate` using SQL CTEs to provide crucial contextual signal.
    * Strictly excluded post-purchase metrics (like `actual_days_to_deliver`) from the training set to prevent Target Data Leakage.
* **Pipeline Implementation:** * Built a defensive `scikit-learn` ColumnTransformer with global median/frequent imputation to guarantee the pipeline will not crash on live, missing production data.
    * Utilized `XGBoost` for its native Sparsity-Aware Split Finding and handled the severe class imbalance natively using `scale_pos_weight` rather than computationally expensive synthetic data generation (SMOTE).
* **Optimization & Serialization:** * Applied `RandomizedSearchCV` to optimize the tree architecture (`max_depth`, `learning_rate`).
    * Serialized the entire end-to-end preprocessing and prediction pipeline using `joblib` (`.pkl`) to ensure the Streamlit UI can cleanly pass raw JSON data directly to the model for real-time inference.
  
## Date: June 2026 (Predictive Engine Expansion)
**Phase 7: Unsupervised Churn & Risk Profiling (RFM)**
* **Architectural Challenge:** The Olist dataset lacks an explicit "deleted account" label, making supervised classification impossible for churn prediction.
* **Pipeline Implementation:** Engineered an unsupervised K-Means clustering pipeline utilizing RFM (Recency, Frequency, Monetary) metrics fused with `avg_review_score` to group customers into behavioral risk profiles.
* **MLOps Resiliency (Dynamic Profiling):** Eliminated the risk of "Cluster Shuffling" (where K-Means randomly assigns cluster IDs 0, 1, 2 on retraining). Engineered dynamic mapping logic (`idxmin()` and `idxmax()`) that mathematically evaluates cluster centers at runtime (e.g., assigning "High Risk" to the cluster with the lowest average review score) before serializing the pipeline. 

**Phase 8: High-Dimensional Time-Series Forecasting**
* **Algorithm Selection:** Selected `XGBRegressor` over traditional univariate models (ARIMA, Prophet) to build a Single Global Model capable of forecasting revenue and inventory across 50+ distinct product categories simultaneously.
* **Feature Engineering (Memory Injection):** Tree-based models treat rows independently. To inject authentic time-series memory, engineered explicit Lag Features (`shift(1)`) and Exponentially Weighted Moving Averages (EWMA), forcing the model to respect the chronological decay of recent momentum.
* **Data Sparsity Resolution:** Encountered catastrophic error rates on raw daily data due to zero-inflated, highly volatile "Long Tail" category sales. Refactored the data ingestion to resample into continuous Weekly buckets (`resample('W')`), absorbing zero-days and smoothing massive outliers to make forecasting mathematically feasible.
* **Segmented Evaluation Strategy:** Discarded standard Global MAE metrics, which were heavily skewed by low-volume noise categories. Implemented a Segmented Evaluation script to isolate and measure the "Top 5 Giants," proving a highly successful baseline relative error rate of ~35% prior to the future integration of exogenous variables (marketing spend, promotions).

**Phase 9: Evaluation & Temporal Anchoring**
* **Automated RAG Evaluation (LLM-as-a-Judge):** Bypassed the manual data-labeling bottleneck of traditional MRR/nDCG metrics. Engineered a custom automated evaluation script (evaluate_rag.py) utilizing GPT-4o to score the RAG pipeline on a strict Triad: Context Relevance, Faithfulness, and Answer Relevance. Achieved a 9.2/10 Faithfulness score, mathematically proving hallucination resistance.
* **Temporal Misalignment Resolution:** Discovered the LLM text-to-SQL engine was failing on relative time queries ("last quarter") because it defaulted to the present-day system clock against a 2018 historical database. Engineered a dynamic system prompt anchor using MAX(order_date) to dynamically synchronize the LLM's internal clock to the dataset's latest boundary, instantly resolving temporal hallucination.