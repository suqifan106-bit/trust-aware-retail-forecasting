# Data

This project uses the Kaggle competition **Walmart Recruiting – Store Sales Forecasting**:
<https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data>

The files are not redistributed in this repository. Accept the competition rules on Kaggle, then download them either manually or with the Kaggle API:

```bash
pip install kaggle                      # requires ~/.kaggle/kaggle.json
kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data/raw
cd data/raw && unzip -o walmart-recruiting-store-sales-forecasting.zip
unzip -o train.csv.zip && unzip -o features.csv.zip   # stores.csv is not zipped
```

Expected files in this folder:

| File | Rows | Used columns |
|---|---|---|
| `train.csv` | 421,570 | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `features.csv` | 8,190 | Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday |
| `stores.csv` | 45 | Store, Type, Size |

`test.csv` from the competition is not used (it has no sales values).

MD5 checksums of the files used in the paper. Copies obtained elsewhere may differ in line endings; the row counts above must match.

```
9663f2973887c5fad0ad271fc64a5dbe  train.csv
26d777bfc9c5b86b35dbbbb2d1726748  features.csv
7f9ba591e643062af41888db96c1e3b6  stores.csv
```
