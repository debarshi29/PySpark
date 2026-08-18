# PySpark

Personal practice repo for learning Apache Spark / PySpark — DataFrame operations,
joins, structured streaming, and RDD basics — plus a notebook-based EY training exercise.

## Structure

```
.
├── RKMVERI/          # Core PySpark practice scripts (DataFrames, joins, streaming, RDDs)
├── EY Training/       # Notebook + sample CSV/Excel data for a separate training exercise
├── input/             # Sample CSV data used by the streaming/join scripts
└── requirements.txt
```

### RKMVERI/

| File | Topic |
|---|---|
| `session.py`, `test.py` | Basic SparkSession setup and a minimal socket stream |
| `df_ex.py`, `df_ex_2.py` | DataFrame basics |
| `df_join.py` | Split / explode / join on DataFrames |
| `join.py`, `join2.py`, `join_filter.py`, `exam.py` | Stream-to-stream joins with watermarking |
| `stream_event.py`, `stream_file.py`, `stream_jf.py`, `stream_join.py` | Structured Streaming: socket & file sources, join + filter, static-stream join |
| `practice_classes.py`, `practice_teachers.py` | DataFrame practice (reads local CSVs — update the hardcoded `path` before running) |
| `rddwc.py` | RDD word count |
| `text_writer.py` | Small utility script for generating sample text |
| `java_version_switch.txt` | PowerShell snippet for switching the active `JAVA_HOME` |

Scripts using `socket` streaming sources expect a listener on the given port, e.g.:

```powershell
# in a separate terminal, feed lines to the stream
ncat -lk 9999
```

Scripts reading from `input/events/` or `input/participants/` expect to be run from the
repo root so the relative path resolves.

### EY Training/

A Jupyter notebook (`pyspark_practice.ipynb`) plus its sample data
(`retail_orders.csv`, `dim_products.csv`, `dim_reps.csv`, `dim_targets.csv`,
`excel_practice.xlsx`).

## Setup

Requires Python 3.10+ and a JDK (Java 8/11/17/21 all work with Spark 4.x).

```powershell
python -m venv venv310
venv310\Scripts\pip install -r requirements.txt
```

Register the venv as a Jupyter kernel:

```powershell
venv310\Scripts\python -m ipykernel install --user --name pyspark-venv --display-name "PySpark (venv310)"
```

## Running

Scripts:

```powershell
venv310\Scripts\python RKMVERI\df_join.py
```

Notebooks:

```powershell
venv310\Scripts\jupyter lab
```
then select the **PySpark (venv310)** kernel.

## Quick sanity check

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("sanity-check").getOrCreate()
spark.range(5).show()
spark.stop()
```
