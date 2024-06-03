# VIX
This is a project directory regarding demonstration of recreating VIX index following CBOE's manual and methodology within period of 2019/01/02 ~ 2019/06/28.

In total 6 files are available here:
**Demonstration**
- _README.md: Also known as this file, serves as a demonstration of this directory.
**Code**
- VIX.ipynb: code file in a markdown-display fashion. I added some traditional Chinese descriptions and comments within this script, which might arise encoding issue if you were trying to understand the comments. In this case, 'UTF-8' is the encoding style that works.
- VIX.py: code file: classic python script, yet traditional Chinese comments exist within this script. If you're only interested in the code then feel free to ignore them.
**Documentations**
- Cboe_Volatility_Index_Mathematics_Methodology.pdf: [source](https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf)
- Johnson-RiskPremiaVIX-2017.pdf: : [source](https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Volatility_Index_Mathematics_Methodology.pdf)
- Volatility_Index_Methodology_Cboe_Volatility_Index.pdf: [source](https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Volatility_Index_Mathematics_Methodology.pdf)
**Data**
- VIX related data.zip
|- ^VIX.csv: historical VIX market data during the period 2019/01/02 ~ 2019/06/28. [source](https://www.investing.com/indices/us-spx-500-historical-data)
|- 2019UST_Yield.csv: historical UST yield curve during the period 2019/01/02 ~ 2019/06/28. [source](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202405)
|- option_price.csv: option data from Wrds. [source](https://wrds-www.wharton.upenn.edu/)
|- vixts.csv: output of (Johnson 2017)'s methodology. I got this data from course lecturer, guess it's from the author of the (Johnson 2017) paper.
