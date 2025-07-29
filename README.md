# CTIC Economic Twitter Project

This repository contains code and resources for analyzing Twitter data with two main tasks:
1. **Number Detection**: Detect whether a tweet contains any number or numeric format.
2. **Topic Assignment**: Assign relevant topics to each tweet using a predefined set of political and social topics.

---

## Project Structure

```bash
CTIC-economic-twitter-project/
├── data/                # Input Twitter data (.xlsx files)
├── prompts/             # Prompt templates for each task
│   ├── number_detection_prompt.txt
│   └── topic_prompt.txt
├── results/             # Output folder for detection results
├── results_0403/        # Output folder for topic results
├── num_dect_run.sh      # Shell script to run number detection
├── topic_run.sh         # Shell script to run topic assignment
├── number_detection.py  # Main script for number detection
├── topic_assignment.py  # Main script for topic assignment
├── utils.py             # Utility functions
```

---

## 1. Number Detection Task

**Goal:**  
Detect if a tweet contains any numbers or numeric formats (e.g., '3', 'three-star', '2024').

- For each tweet, the script outputs `1` if a number is detected, `0` otherwise.
- Results are saved in an Excel file.

**Prompt Example (`prompts/number_detection_prompt.txt`)**

**To run (example):**
```bash
bash num_dect_run.sh
```
Edit ``num_dect_run.sh`` to set your API key, input, and output file paths as needed.

---
## 2. Topic Assignment Task

**Goal:**  
Assign one or more topics from a defined set to each tweet, based on content.

- Topic include: health_exp, public_assist, immigration_boarder, abortion, LGBTQ, international_relation, domestic_crime, inflation_live_exp, religion, unemployment, income_tax, middle_class, state_name, competitor_name, supporter_name, democracy, pluralist_values, disagreement, republican, democratic. 

**Each topic is defined with examples in (`prompts/topic_prompt.txt`)**

**To run (example):**
```bash
bash topic_run.sh
```
Edit ``topic_run.sh`` to set your API key, input, and output file paths as needed.

---

## Folders

- `data/`: Place your input Twitter .xlsx files here (e.g., `test.xlsx`).
- `results/`: Contains the results.
- `prompts/`: Contains all prompt templates for each task.

---

## Usage Notes

- Both tasks require OpenAI API access (API_KEY), set in the respective `.sh` scripts.

- You can specify which model to use (`gpt-4o`, `gpt-4`, etc.) in the `.sh` scripts.

- For partial processing, set NUM_SAMPLES to a specific number; -1 processes all data.

---

## Getting Started

1. Clone this repo
2. Place your Twitter data in `data/` (as Excel files)
3. Edit the `.sh` scripts to set API keys and file paths
4. Run the scripts to generate results

## Contact
For questions or contributions, please contact `luxinyuan@u.nus.edu`