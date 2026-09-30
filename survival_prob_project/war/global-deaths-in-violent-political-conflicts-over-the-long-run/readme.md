# Global deaths in violent political conflicts over the long run - Data package

This data package contains the data that powers the chart ["Global deaths in violent political conflicts over the long run"](https://ourworldindata.org/grapher/global-deaths-in-violent-political-conflicts-over-the-long-run?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on November 24, 2025.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Deaths in ongoing conflicts - Conflict type: all
The estimated number of deaths in ongoing armed conflicts that year.
Last updated: September 21, 2023  
Date range: 1400–2000  
Unit: deaths  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Brecke - Conflict Catalog (1999) – with major processing by Our World in Data

#### Full citation
Brecke - Conflict Catalog (1999) – with major processing by Our World in Data. “Deaths in ongoing conflicts - Conflict type: all” [dataset]. Brecke, “Conflict Catalog” [original data].
Source: Brecke - Conflict Catalog (1999) – with major processing by Our World In Data

### What you should know about this data
* The data has been collated from a range of historical studies by the political scientist Peter Brecke.
* Brecke seeks to include only political disagreements between organized groups, or between an organized group and civilians, that cause at least 32 deaths during a year. This definition tries to exclude interpersonal and criminal violence.
* His data seeks to include combatant and civilian deaths, counting both direct deaths, due to fighting, and indirect deaths due to disease and starvation.
* The value of this data lies in giving us some insight into conflicts over the very long-run. But there are important limitations. The historical studies on which this data relies vary greatly in terms of how deaths are counted and the reliability of the estimates. Moreover, as Brecke makes clear, the data he has been able to collect provides an incomplete record – particularly for the distant past and outside of Europe.

### Source

#### Brecke – Conflict Catalog
Retrieved on: 2023-09-21  
Retrieved from: https://brecke.inta.gatech.edu/research/conflict/  

#### Notes on our processing step for this indicator
Brecke's catalog provides data per conflict. In order to create an annual time series, we sum the deaths recorded in each year across all conflicts, distributing deaths evenly across the duration of conflicts lasting more than one year.

The catalog includes many conflicts for which Brecke was unable to find a deaths estimate. Given Brecke's stated definition of a conflict – political disagreements between organized groups, or between an organized group and civilians, that cause at least 32 deaths during a year – we assign such conflicts 32 deaths when creating the time series.


    