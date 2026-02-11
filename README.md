# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Udaiveer Singh  
**Roll Number:** U20230017  
**GitHub Branch:** udaiveer_U20230017

---

## Table of Contents
1. [Overview](#overview)
2. [Approach and Design Decisions](#approach-and-design-decisions)
3. [Key Results and Observations](#key-results-and-observations)
4. [Reproduction Instructions](#reproduction-instructions)
5. [File Structure](#file-structure)
6. [Dependencies](#dependencies)

---

## Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** framework for recommending news articles to users. The system treats user categories as contexts and news categories as bandit arms, learning optimal policies to maximize user engagement.

The implementation consists of three main components:
1. **User Classification** - Predicts user type (User1, User2, User3) from feature data
2. **Contextual Bandit Algorithms** - Three exploration-exploitation strategies (Epsilon-Greedy, UCB, SoftMax)
3. **Recommendation Engine** - End-to-end pipeline that classifies users and recommends articles

---

## Approach and Design Decisions

### 1. Data Preprocessing (10 Points)

**Strategy:**
- **Missing Values:** Age column had ~35% missing values. Imputed using median strategy to preserve distribution characteristics.
- **Feature Encoding:** 
  - Categorical features (region_code, browser_version) encoded using LabelEncoder
  - Boolean features (subscriber) converted to integer (0/1)
  - String columns (user_id) excluded from model features
- **Feature Selection:** Removed identifier columns (user_id) and retained 31 features for classification

**Design Rationale:**
Median imputation for age was chosen over mean to avoid sensitivity to outliers in user demographics. Label encoding was sufficient for tree-based and linear models as there was no ordinal relationship assumed.

---

### 2. User Classification (10 Points)

**Model Selection:** Logistic Regression with multi-class support
- **Algorithm:** `LogisticRegression(max_iter=1000, random_state=42)`
- **Training Split:** 80% training (1600 samples), 20% validation (400 samples)
- **Performance:** Achieved 82.75% validation accuracy

**Design Rationale:**
Logistic Regression was selected for its:
- Strong baseline performance for multi-class problems
- Fast training and inference (important for real-time recommendation)
- Interpretable coefficients for understanding user segments
- Robust performance without extensive hyperparameter tuning

**Classification Report Summary:**
```
              precision    recall  f1-score   support
     user_1       0.86      0.80      0.83       146
     user_2       0.78      0.85      0.81       124
     user_3       0.84      0.84      0.84       130
   accuracy                           0.83       400
```

The classifier showed balanced performance across all three user types, making it reliable as the context detector for the bandit system.

---

### 3. Contextual Bandit Implementation (45 Points)

The core challenge was mapping 12 arms (3 contexts × 4 news categories) to the reward sampler, following the specification:

| Arm Index (j) | News Category | User Context |
|--------------|---------------|--------------|
| 0-3          | Entertainment, Education, Tech, Crime | User1 |
| 4-7          | Entertainment, Education, Tech, Crime | User2 |
| 8-11         | Entertainment, Education, Tech, Crime | User3 |

#### 3.1 Epsilon-Greedy (15 Points)

**Algorithm:**
- For each context, maintain Q-values (estimated mean rewards) and visit counts for 4 arms
- With probability ε: explore (random arm)
- With probability 1-ε: exploit (arm with highest Q-value)
- Update Q-values using incremental averaging: Q_new = Q_old + α(reward - Q_old)

**Hyperparameter Testing:**
- Tested ε ∈ {0.05, 0.1, 0.2}
- **Finding:** ε = 0.1 provided best balance
  - ε = 0.05: Converged quickly but potentially suboptimal (premature exploitation)
  - ε = 0.1: Good exploration-exploitation trade-off, highest final rewards
  - ε = 0.2: Too much exploration, slower convergence

**Design Decision:**
Fixed learning rate α = 0.1 was used for Q-value updates to give recent rewards appropriate weight while maintaining stability.

---

#### 3.2 Upper Confidence Bound - UCB (15 Points)

**Algorithm:**
- UCB action selection: Select arm with max(Q_i + C × √(ln(t) / n_i))
- Q_i: estimated reward for arm i
- n_i: number of times arm i was selected
- t: total timesteps
- C: exploration parameter

**Hyperparameter Testing:**
- Tested C ∈ {0.5, 1.0, 2.0}
- **Finding:** C = 1.0 achieved best performance
  - C = 0.5: Insufficient exploration, got stuck in local optima
  - C = 1.0: Optimal balance, fastest convergence to best arms
  - C = 2.0: Over-exploration, slower convergence

**Design Rationale:**
UCB's confidence bound naturally decreases as arms are tried more often, providing principled exploration that adapts over time. This is theoretically superior to fixed ε strategies.

---

#### 3.3 SoftMax (15 Points)

**Algorithm:**
- Probability of selecting arm i: P(i) = exp(Q_i / τ) / Σ_j exp(Q_j / τ)
- τ (temperature): Fixed at 1.0 as per assignment specification
- Higher Q-values → higher selection probability (but never zero)

**Performance:**
- Provided smooth, probabilistic exploration
- More stable than epsilon-greedy (no sudden random jumps)
- Slightly slower convergence than UCB
- Good when reward distributions have high variance

**Design Rationale:**
Temperature τ = 1.0 provides moderate exploration. Higher τ increases exploration (flatter distribution), lower τ increases exploitation (sharper distribution).

---

### 4. Recommendation Engine (20 Points)

**System Architecture:**

```
User Features → User Classifier → User Context (User1/2/3)
                                       ↓
                            Bandit Policy (ε-greedy/UCB/SoftMax)
                                       ↓
                            Selected News Category (Entertainment/Education/Tech/Crime)
                                       ↓
                            Sample Random Article from Category
```

**Implementation Challenges:**

1. **Category Mapping Issue:** The dataset's actual news categories didn't match the abstract bandit categories (Entertainment, Education, Tech, Crime). 
   
   **Solution:** Created semantic mapping:
   ```python
   CATEGORY_MAP = {
       "Entertainment": ["ENTERTAINMENT", "SPORTS", "WELLNESS"],
       "Education": ["EDUCATION", "SCIENCE"],
       "Tech": ["TECH", "BUSINESS"],
       "Crime": ["CRIME", "POLITICS"]
   }
   ```

2. **Fallback Strategy:** If no articles match the selected category, randomly sample from entire dataset to ensure system robustness.

**End-to-End Workflow:**
```python
def recommend_article(user_row, bandits):
    # Step 1: Classify user
    user_context = predict_user_context(user_row)  # "user_1", "user_2", or "user_3"
    
    # Step 2: Select best category using bandit
    category = select_best_category(bandits, user_context)  # "Entertainment", etc.
    
    # Step 3: Sample article from that category
    article = sample_article(news_df, category)
    
    return {
        "predicted_user_context": user_context,
        "recommended_category": category,
        "article_id": article["article_id"],
        "article_title": article["title"]
    }
```

---

## Key Results and Observations

### Classification Performance
- **Validation Accuracy:** 82.75%
- **Balanced Performance:** F1-scores ranged from 0.81-0.84 across all user types
- **No Significant Bias:** Model performed equally well on all user contexts

### Bandit Algorithm Comparison

**Simulation Parameters:** T = 10,000 timesteps

| Algorithm | Final Avg Reward | Convergence Speed | Stability |
|-----------|------------------|-------------------|-----------|
| **UCB (C=1.0)** | **Highest** | Fast (< 2000 steps) | Very Stable |
| **Epsilon-Greedy (ε=0.1)** | High | Medium (3000 steps) | Moderate variance |
| **SoftMax (τ=1.0)** | Medium-High | Slow (4000 steps) | Very Stable |

### Context-Specific Insights

**User1 Context:**
- Best arm: Crime category (highest rewards)
- UCB identified this fastest (~1500 steps)
- Epsilon-greedy showed more variance but eventually converged

**User2 Context:**
- Best arm: Crime category
- All algorithms converged similarly
- Lower overall rewards than User1

**User3 Context:**
- Best arm: Tech category
- SoftMax showed smoother learning curve
- UCB still fastest to converge

### Hyperparameter Sensitivity

**Epsilon (ε) Effects:**
- **ε = 0.05:** Fast convergence but risk of suboptimal policy
- **ε = 0.10:** Best overall performance (recommended)
- **ε = 0.20:** Too exploratory, sacrifices short-term rewards

**UCB Constant (C) Effects:**
- **C = 0.5:** Insufficient exploration, 5-10% lower final rewards
- **C = 1.0:** Optimal balance (recommended)
- **C = 2.0:** Over-exploration, 15-20% slower convergence

### Key Findings

1. **UCB Superiority:** UCB consistently outperformed other methods due to its adaptive, confidence-based exploration strategy.

2. **Context Matters:** Different user types showed distinct preferences for news categories, validating the contextual bandit approach over a single global policy.

3. **Exploration-Exploitation Trade-off:** Proper hyperparameter tuning is critical. Too little exploration risks missing the best arm; too much delays convergence.

4. **Practical Considerations:** 
   - UCB recommended for production (best performance)
   - Epsilon-greedy acceptable with ε=0.1 (simpler, interpretable)
   - SoftMax useful when reward variance is high (smoother updates)

---

## Reproduction Instructions

### Prerequisites
- Python 3.12 or higher
- Git
- pip package manager

### Step 1: Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/udaiveersingh/lab3-contextual-bandit.git
cd lab3-contextual-bandit

# Checkout the correct branch
git checkout udaiveer_U20230017
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib scikit-learn

# Install the custom sampler package
pip install rlcmab-sampler
```

### Step 3: Verify Data Files

Ensure the following data files are present in the `data/` directory:
- `train_users.csv`
- `test_users.csv`
- `news_articles.csv`

### Step 4: Run the Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open lab3_results_17.ipynb
# Execute all cells: Cell → Run All
```

**Expected Runtime:** 5-10 minutes (depending on machine)

### Step 5: Verify Outputs

The notebook should produce:
-  Classification report with ~82% accuracy
-  9 reward vs. time plots (3 contexts × 3 algorithms)
-  Hyperparameter comparison plots for ε and C
-  Sample recommendations for test users

### Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'rlcmab_sampler'`
- **Solution:** Run `pip install rlcmab-sampler`

**Issue:** Data files not found
- **Solution:** Ensure you're in the repository root directory and `data/` folder exists

**Issue:** Plots not displaying
- **Solution:** Run `%matplotlib inline` in a notebook cell

**Issue:** Different results on re-run
- **Solution:** Random seeds are set (random_state=42), but sampler outputs may vary

---

## File Structure

```
lab3-contextual-bandit/
│
├── lab3_results_17.ipynb       # Main submission notebook
├── README.md                   # This file
├── assignment.pdf              # Assignment specifications
│
└── data/
    ├── train_users.csv         # Training user data (2000 samples)
    ├── test_users.csv          # Test user data
    └── news_articles.csv       # News article dataset
```

---

## Dependencies

### Core Libraries
- **numpy** (1.24+): Numerical computations, array operations
- **pandas** (2.0+): Data manipulation and preprocessing
- **matplotlib** (3.7+): Visualization and plotting

### Machine Learning
- **scikit-learn** (1.3+): Classification models and metrics
  - `LogisticRegression`: User classification
  - `train_test_split`: Data splitting
  - `LabelEncoder`: Categorical encoding
  - `classification_report`: Model evaluation

### Custom Package
- **rlcmab-sampler**: Provided reward distribution sampler (do not modify)

### Python Version
- **Python 3.12+** required

---

## Algorithm Pseudocode

### Epsilon-Greedy
```
Initialize Q[context][arm] = 0, N[context][arm] = 0
For t = 1 to T:
    Observe context c
    With probability ε:
        Select random arm a
    With probability 1-ε:
        Select a = argmax_i Q[c][i]
    
    Receive reward r from sampler(context_offset + a)
    N[c][a] += 1
    Q[c][a] += (r - Q[c][a]) / N[c][a]
```

### UCB
```
Initialize Q[context][arm] = 0, N[context][arm] = 0
For t = 1 to T:
    Observe context c
    For each arm i:
        UCB[i] = Q[c][i] + C × sqrt(ln(t) / max(N[c][i], 1))
    
    Select a = argmax_i UCB[i]
    Receive reward r from sampler(context_offset + a)
    N[c][a] += 1
    Q[c][a] += (r - Q[c][a]) / N[c][a]
```

### SoftMax
```
Initialize Q[context][arm] = 0, N[context][arm] = 0
For t = 1 to T:
    Observe context c
    For each arm i:
        P[i] = exp(Q[c][i] / τ) / Σ_j exp(Q[c][j] / τ)
    
    Sample a ~ P
    Receive reward r from sampler(context_offset + a)
    N[c][a] += 1
    Q[c][a] += (r - Q[c][a]) / N[c][a]
```