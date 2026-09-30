# Death rate in wars - Data package

This data package contains the data that powers the chart ["Death rate in wars"](https://ourworldindata.org/grapher/death-rate-in-wars-project-mars?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website.

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


## Civil wars
The low estimate of the number of deaths in ongoing conventional civil wars that year.
Last updated: September 21, 2023  
Date range: 1816–2011  
Unit: deaths per 100,000 people  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Jason Lyall, Project Mars (2020); Population based on various sources (2023) – with major processing by Our World in Data

#### Full citation
Jason Lyall, Project Mars (2020); Population based on various sources (2023) – with major processing by Our World in Data. “Civil wars” [dataset]. Jason Lyall, “Project Mars”; Various sources, “Population” [original data].
Source: Jason Lyall, Project Mars (2020), Population based on various sources (2023) – with major processing by Our World In Data

### What you should know about this data
* A civil war is defined by [Project Mars](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DUO7IE) as a conflict between combatants that were previously part of the same state, but now at least one seeks control or secession. It is fought with differentiated militaries and clear frontlines and causes at least 500 deaths over its duration.
* If a war lasted more than one year, we distributed its deaths evenly across years.
* Project Mars identifies war deaths based on contemporary sources and academic research.
* Project Mars considers the 1982-1984 war in Lebanon and the Afghan War from 1996-2001 to have had elements of both an interstate and a civil war. We have treated them overall as interstate wars to make our charts simpler.
* The conflict-level data by Project Mars cannot be neatly mapped to individual countries, which is why we only present aggregated data for the world and world regions.

### Sources

#### Jason Lyall – Project Mars
Retrieved on: 2023-09-21  
Retrieved from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DUO7IE  

#### Various sources – Population
Retrieved on: 2023-03-31  
Retrieved from: https://ourworldindata.org/population-sources  


## Interstate wars
The low estimate of the number of deaths in ongoing conventional interstate wars that year.
Last updated: September 21, 2023  
Date range: 1816–2011  
Unit: deaths per 100,000 people  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Jason Lyall, Project Mars (2020); Population based on various sources (2023) – with major processing by Our World in Data

#### Full citation
Jason Lyall, Project Mars (2020); Population based on various sources (2023) – with major processing by Our World in Data. “Interstate wars” [dataset]. Jason Lyall, “Project Mars”; Various sources, “Population” [original data].
Source: Jason Lyall, Project Mars (2020), Population based on various sources (2023) – with major processing by Our World In Data

### What you should know about this data
* An interstate war is defined by [Project Mars](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DUO7IE) as a conflict between states with differentiated militaries and clear frontlines that causes at least 500 deaths over its duration.
* If a war lasted more than one year, we distributed its deaths evenly across years.
* Project Mars identifies war deaths based on contemporary sources and academic research.
* Project Mars considers the 1982-1984 war in Lebanon and the Afghan War from 1996-2001 to have had elements of both an interstate and a civil war. We have treated them overall as interstate wars to make our charts simpler.
* The conflict-level data by Project Mars cannot be neatly mapped to individual countries, which is why we only present aggregated data for the world and world regions.

### Sources

#### Jason Lyall – Project Mars
Retrieved on: 2023-09-21  
Retrieved from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DUO7IE  

#### Various sources – Population
Retrieved on: 2023-03-31  
Retrieved from: https://ourworldindata.org/population-sources  


    