# DATA 210P Final Project: Predicting Anime Success

A statistical analysis of MyAnimeList data to predict anime ratings using linear regression, logistic regression, and random forest models.

## 📁 Project Structure

```
anime_project/
├── data/
│   ├── anime-dataset-2023.csv      # Raw data (download from Kaggle)
│   └── anime_modeling_data.csv     # Prepared data (generated)
├── outputs/
│   ├── linear_regression_diagnostics.pdf
│   ├── logistic_regression_results.pdf
│   └── model_comparison.pdf
├── 01_data_preparation.py          # Step 1: Clean and prepare data
├── 02_linear_regression.py         # Step 2: Linear regression models
├── 03_logistic_regression.py       # Step 3: Logistic regression (Day 2)
├── 04_random_forest.py             # Step 4: Random forest (Day 3)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/anime-success-prediction.git
cd anime-success-prediction
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Data

1. Go to [Kaggle: MyAnimeList Dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
2. Download `anime-dataset-2023.csv`
3. Place it in the `data/` folder

### 5. Run the Analysis

```bash
# Step 1: Prepare the data
python 01_data_preparation.py

# Step 2: Run linear regression
python 02_linear_regression.py

# Step 3: Run logistic regression (coming Day 2)
python 03_logistic_regression.py

# Step 4: Run random forest (coming Day 3)
python 04_random_forest.py
```

## 📊 Research Questions

1. **Primary (Linear Regression):** What factors predict anime rating scores?
2. **Secondary (Logistic Regression):** Can we predict if an anime will be highly rated (≥8.0)?
3. **Tertiary (Random Forest):** How do flexible models compare to traditional regression?

## 📈 Key Findings

- **R² = 0.40**: The model explains 40% of variance in anime scores
- **Top positive predictors:** Drama genre, Adventure, longer episode counts
- **Top negative predictors:** ONA type, Original source (vs. Manga adaptation)
- **Missing data:** MAR mechanism - unpopular anime lack scores

## 📚 Dataset

- **Source:** MyAnimeList (2023)
- **Size:** 24,905 anime (15,596 with valid scores)
- **Outcome:** Score (1-10 scale)
- **Predictors:** Type, Source, Genres, Studios, Episodes, Year

## 🛠️ Technologies

- Python 3.10+
- pandas, numpy
- statsmodels (regression)
- scikit-learn (random forest)
- matplotlib, seaborn (visualization)

## 📝 Author

Rohan - DATA 210P: Statistical Methods I, Winter 2026

## 📄 License

This project is for educational purposes.
