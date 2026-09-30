# Death rate in state-based conflicts - Data package

This data package contains the data that powers the chart ["Death rate in state-based conflicts"](https://ourworldindata.org/grapher/death-rate-in-state-based-conflicts?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The remaining columns are the data columns, each of which is a time series. If the CSV data is downloaded using the "full data" option, then each column corresponds to one time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data columns are transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

### How we process data at Our World In Data
All data and visualizations on Our World in Data rely on data sourced from one or several original data providers. Preparing this original data involves several processing steps. Depending on the data, this can include standardizing country names and world region definitions, converting units, calculating derived indicators such as per capita measures, as well as adding or adapting metadata such as the name or the description given to an indicator.
[Read about our data pipeline](https://docs.owid.io/projects/etl/)

## Detailed information about each time series


## Extrasystemic
The best estimate of the number of deaths of combatants and civilians due to fighting in extrasystemic conflicts that were ongoing that year.
Last updated: June 16, 2025  
Next update: June 2026  
Date range: 1946–2024  
Unit: deaths per 100,000 people  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Uppsala Conflict Data Program (2025) and other sources – with minor processing by Our World in Data

#### Full citation
Uppsala Conflict Data Program (2025); geoBoundaries (2023); Peace Research Institute Oslo (2017); Population based on various sources (2024) – with minor processing by Our World in Data. “Extrasystemic” [dataset]. Uppsala Conflict Data Program, “Georeferenced Event Dataset v25.1”; geoBoundaries, “geoBoundaries - Comprehensive Global Administrative Zones (CGAZ) 6.0.0”; Peace Research Institute Oslo, “Battle deaths v3.1”; Various sources, “Population” [original data].
Source: Uppsala Conflict Data Program (2025), geoBoundaries (2023), Peace Research Institute Oslo (2017), Population based on various sources (2024) – with minor processing by Our World In Data

### What you should know about this data
* A colonial conflict is defined by the [Uppsala Conflict Data Program (UCDP)](https://ucdp.uu.se/) and the [Peace Research Institute Oslo (PRIO)](https://www.prio.org/data/1) as a conflict between a state and a non-state armed group that causes at least 25 deaths during a year, and takes place outside of the state's territory. This includes combatant and civilian deaths due to fighting.
* [UCDP](https://www.uu.se/en/department/peace-and-conflict-research/research/ucdp/ucdp-methodology) and [PRIO](https://www.semanticscholar.org/paper/Monitoring-Trends-in-Global-Combat%3A-A-New-Dataset-Lacina-Gleditsch/0c4ff27fff986bc30112ac59ad6afbd7b719ad17?p2df) identify conflict deaths based on news reports, other contemporary sources, and academic research.
* PRIO's conflict-level data cannot be neatly mapped to individual countries, which is why we only present aggregated data for the world and world regions.
* We show here the "best" death estimates as identified by UCDP and PRIO. They also report high and low estimates.

### Sources

#### Uppsala Conflict Data Program – Georeferenced Event Dataset
Retrieved on: 2025-06-13  
Retrieved from: https://ucdp.uu.se/downloads/index.html#ged_global  

#### geoBoundaries – geoBoundaries - Comprehensive Global Administrative Zones (CGAZ)
Retrieved on: 2025-06-26  
Retrieved from: https://www.geoboundaries.org/globalDownloads.html  

#### Peace Research Institute Oslo – Battle deaths
Retrieved on: 2023-09-21  
Retrieved from: https://www.prio.org/data/1  

#### Various sources – Population
Retrieved on: 2024-07-11  
Retrieved from: https://ourworldindata.org/population-sources  

#### Notes on our processing step for this indicator
Data prior to 1989 is sourced from PRIO. Data since 1989 is sourced from UCDP.

For conflict years without a best deaths estimate in the PRIO data, we conservatively coded the low estimate.


## Intrastate
The best estimate of the number of deaths of combatants and civilians due to fighting in intrastate conflicts that were ongoing that year.
Last updated: June 16, 2025  
Next update: June 2026  
Date range: 1946–2024  
Unit: deaths per 100,000 people  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Uppsala Conflict Data Program (2025) and other sources – with minor processing by Our World in Data

#### Full citation
Uppsala Conflict Data Program (2025); geoBoundaries (2023); Peace Research Institute Oslo (2017); Population based on various sources (2024) – with minor processing by Our World in Data. “Intrastate” [dataset]. Uppsala Conflict Data Program, “Georeferenced Event Dataset v25.1”; geoBoundaries, “geoBoundaries - Comprehensive Global Administrative Zones (CGAZ) 6.0.0”; Peace Research Institute Oslo, “Battle deaths v3.1”; Various sources, “Population” [original data].
Source: Uppsala Conflict Data Program (2025), geoBoundaries (2023), Peace Research Institute Oslo (2017), Population based on various sources (2024) – with minor processing by Our World In Data

### What you should know about this data
* An intrastate conflict is defined by the [Uppsala Conflict Data Program (UCDP)](https://ucdp.uu.se/) and the [Peace Research Institute Oslo (PRIO)](https://www.prio.org/data/1) as a conflict between a state and a non-state armed group that causes at least 25 deaths during a year. This includes combatant and civilian deaths due to fighting. If a foreign state is involved, it is called "internationalized", and "non-internationalized" otherwise.
* [UCDP](https://www.uu.se/en/department/peace-and-conflict-research/research/ucdp/ucdp-methodology) and [PRIO](https://www.semanticscholar.org/paper/Monitoring-Trends-in-Global-Combat%3A-A-New-Dataset-Lacina-Gleditsch/0c4ff27fff986bc30112ac59ad6afbd7b719ad17?p2df) identify conflict deaths based on news reports, other contemporary sources, and academic research.
* PRIO's conflict-level data cannot be neatly mapped to individual countries, which is why we only present aggregated data for the world and world regions.
* We show here the "best" death estimates as identified by UCDP and PRIO. They also report high and low estimates.

### Sources

#### Uppsala Conflict Data Program – Georeferenced Event Dataset
Retrieved on: 2025-06-13  
Retrieved from: https://ucdp.uu.se/downloads/index.html#ged_global  

#### geoBoundaries – geoBoundaries - Comprehensive Global Administrative Zones (CGAZ)
Retrieved on: 2025-06-26  
Retrieved from: https://www.geoboundaries.org/globalDownloads.html  

#### Peace Research Institute Oslo – Battle deaths
Retrieved on: 2023-09-21  
Retrieved from: https://www.prio.org/data/1  

#### Various sources – Population
Retrieved on: 2024-07-11  
Retrieved from: https://ourworldindata.org/population-sources  

#### Notes on our processing step for this indicator
Data prior to 1989 is sourced from PRIO. Data since 1989 is sourced from UCDP.

For conflict years without a best deaths estimate in the PRIO data, we conservatively coded the low estimate.


## Interstate
The best estimate of the number of deaths of combatants and civilians due to fighting in interstate conflicts that were ongoing that year.
Last updated: June 16, 2025  
Next update: June 2026  
Date range: 1946–2024  
Unit: deaths per 100,000 people  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Uppsala Conflict Data Program (2025) and other sources – with minor processing by Our World in Data

#### Full citation
Uppsala Conflict Data Program (2025); geoBoundaries (2023); Peace Research Institute Oslo (2017); Population based on various sources (2024) – with minor processing by Our World in Data. “Interstate” [dataset]. Uppsala Conflict Data Program, “Georeferenced Event Dataset v25.1”; geoBoundaries, “geoBoundaries - Comprehensive Global Administrative Zones (CGAZ) 6.0.0”; Peace Research Institute Oslo, “Battle deaths v3.1”; Various sources, “Population” [original data].
Source: Uppsala Conflict Data Program (2025), geoBoundaries (2023), Peace Research Institute Oslo (2017), Population based on various sources (2024) – with minor processing by Our World In Data

### What you should know about this data
* An interstate conflict is defined by the [Uppsala Conflict Data Program (UCDP)](https://ucdp.uu.se/) and the [Peace Research Institute Oslo (PRIO)](https://www.prio.org/data/1) as a conflict between states that causes at least 25 deaths during a year. This includes combatant and civilian deaths due to fighting.
* [UCDP](https://www.uu.se/en/department/peace-and-conflict-research/research/ucdp/ucdp-methodology) and [PRIO](https://www.semanticscholar.org/paper/Monitoring-Trends-in-Global-Combat%3A-A-New-Dataset-Lacina-Gleditsch/0c4ff27fff986bc30112ac59ad6afbd7b719ad17?p2df) identify conflict deaths based on news reports, other contemporary sources, and academic research.
* PRIO's conflict-level data cannot be neatly mapped to individual countries, which is why we only present aggregated data for the world and world regions.
* We show here the "best" death estimates as identified by UCDP and PRIO. They also report high and low estimates.

### Sources

#### Uppsala Conflict Data Program – Georeferenced Event Dataset
Retrieved on: 2025-06-13  
Retrieved from: https://ucdp.uu.se/downloads/index.html#ged_global  

#### geoBoundaries – geoBoundaries - Comprehensive Global Administrative Zones (CGAZ)
Retrieved on: 2025-06-26  
Retrieved from: https://www.geoboundaries.org/globalDownloads.html  

#### Peace Research Institute Oslo – Battle deaths
Retrieved on: 2023-09-21  
Retrieved from: https://www.prio.org/data/1  

#### Various sources – Population
Retrieved on: 2024-07-11  
Retrieved from: https://ourworldindata.org/population-sources  

#### Notes on our processing step for this indicator
Data prior to 1989 is sourced from PRIO. Data since 1989 is sourced from UCDP.

For conflict years without a best deaths estimate in the PRIO data, we conservatively coded the low estimate.


    