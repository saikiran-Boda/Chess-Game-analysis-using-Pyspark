# Chess Game Analysis Using PySpark

A comprehensive data analytics pipeline for analyzing over 6 million chess games using PySpark and Python visualization libraries. This project processes large-scale chess game data to extract meaningful patterns and insights about player behavior, opening effectiveness, and game outcomes.

## Features

- Analysis of 6+ million chess games
- Advanced data processing using PySpark
- Comprehensive visualization suite
- Player rating distribution analysis
- Opening effectiveness evaluation
- Game outcome pattern analysis
- Time control preference analysis

## Technologies Used

- Python 3.10+
- PySpark 3.5.3
- Matplotlib
- Seaborn
- Pandas
- Plotly

## Project Structure
```
/
├── src/
│   ├── chess_analysis.py       # Main analysis script
│   └── utils/                  # Utility functions
├── data/
│   └── README.md              # Data directory information
├── results/
│   └── visualizations/        # Generated visualizations
├── docs/
│   └── report.md             # Detailed analysis report
└── requirements.txt          # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/[username]/chess-analysis.git
cd chess-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the chess games dataset and place it in the data directory

4. Run the analysis:
```bash
python src/chess_analysis.py
```

## Key Findings

- Game Outcomes:
  - White wins: 49.8%
  - Black wins: 46.4%
  - Draws: 3.8%

- Opening Success Rates:
  - Italian Game: 52.56% win rate
  - Ruy Lopez: 52.11% win rate
  - English Opening: 51.57% win rate

- Time Control Preferences:
  - Increment: 99.6%
  - Standard: 0.4%

## Visualizations

The project generates several visualizations:
1. Game outcome distribution
2. Opening success rates
3. Player rating distribution
4. Time control preferences
5. Player strength categories

## Contributors

- ~Chenna Keshava
  - PySpark pipeline development
  - Data transformation logic
  - Performance optimization

- Sai Kiran
  - Visualization components
  - Data analysis
  - Documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Kaggle Chess Games Dataset](https://www.kaggle.com/datasets/arevel/chess-games)
- Thanks to the PySpark and Python data science community for their excellent tools and documentation

## Contact

For any queries regarding this project, please open an issue or contact the contributors.
